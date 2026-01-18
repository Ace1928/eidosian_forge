import logging
import logging.config
import os
import sys
import warnings
from typing import List, Type
import click
from redis.exceptions import ConnectionError
from rq import Connection, Retry
from rq import __version__ as version
from rq.cli.helpers import (
from rq.contrib.legacy import cleanup_ghosts
from rq.defaults import (
from rq.exceptions import InvalidJobOperationError
from rq.job import Job, JobStatus
from rq.logutils import blue
from rq.registry import FailedJobRegistry, clean_registries
from rq.serializers import DefaultSerializer
from rq.suspension import is_suspended
from rq.suspension import resume as connection_resume
from rq.suspension import suspend as connection_suspend
from rq.utils import get_call_string, import_attribute
from rq.worker import Worker
from rq.worker_pool import WorkerPool
from rq.worker_registration import clean_worker_registry
@main.command()
@click.option('--burst', '-b', is_flag=True, help='Run in burst mode (quit after all work is done)')
@click.option('--logging-level', '-l', type=str, default='INFO', help='Set logging level')
@click.option('--sentry-ca-certs', envvar='RQ_SENTRY_CA_CERTS', help='Path to CRT file for Sentry DSN')
@click.option('--sentry-debug', envvar='RQ_SENTRY_DEBUG', help='Enable debug')
@click.option('--sentry-dsn', envvar='RQ_SENTRY_DSN', help='Report exceptions to this Sentry DSN')
@click.option('--verbose', '-v', is_flag=True, help='Show more output')
@click.option('--quiet', '-q', is_flag=True, help='Show less output')
@click.option('--log-format', type=str, default=DEFAULT_LOGGING_FORMAT, help='Set the format of the logs')
@click.option('--date-format', type=str, default=DEFAULT_LOGGING_DATE_FORMAT, help='Set the date format of the logs')
@click.option('--job-class', type=str, default=None, help='Dotted path to a Job class')
@click.argument('queues', nargs=-1)
@click.option('--num-workers', '-n', type=int, default=1, help='Number of workers to start')
@pass_cli_config
def worker_pool(cli_config, burst: bool, logging_level, queues, serializer, sentry_ca_certs, sentry_debug, sentry_dsn, verbose, quiet, log_format, date_format, worker_class, job_class, num_workers, **options):
    """Starts a RQ worker pool"""
    settings = read_config_file(cli_config.config) if cli_config.config else {}
    queue_names: List[str] = queues or settings.get('QUEUES', ['default'])
    sentry_ca_certs = sentry_ca_certs or settings.get('SENTRY_CA_CERTS')
    sentry_debug = sentry_debug or settings.get('SENTRY_DEBUG')
    sentry_dsn = sentry_dsn or settings.get('SENTRY_DSN')
    setup_loghandlers_from_args(verbose, quiet, date_format, log_format)
    if serializer:
        serializer_class: Type[DefaultSerializer] = import_attribute(serializer)
    else:
        serializer_class = DefaultSerializer
    if worker_class:
        worker_class = import_attribute(worker_class)
    else:
        worker_class = Worker
    if job_class:
        job_class = import_attribute(job_class)
    else:
        job_class = Job
    pool = WorkerPool(queue_names, connection=cli_config.connection, num_workers=num_workers, serializer=serializer_class, worker_class=worker_class, job_class=job_class)
    pool.start(burst=burst, logging_level=logging_level)
    if sentry_dsn:
        sentry_opts = {'ca_certs': sentry_ca_certs, 'debug': sentry_debug}
        from rq.contrib.sentry import register_sentry
        register_sentry(sentry_dsn, **sentry_opts)