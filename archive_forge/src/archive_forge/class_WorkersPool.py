import os
import sys
import click
from click import ParamType
from click.types import StringParamType
from celery import concurrency
from celery.bin.base import (COMMA_SEPARATED_LIST, LOG_LEVEL, CeleryDaemonCommand, CeleryOption,
from celery.concurrency.base import BasePool
from celery.exceptions import SecurityError
from celery.platforms import EX_FAILURE, EX_OK, detached, maybe_drop_privileges
from celery.utils.log import get_logger
from celery.utils.nodenames import default_nodename, host_format, node_format
class WorkersPool(click.Choice):
    """Workers pool option."""
    name = 'pool'

    def __init__(self):
        """Initialize the workers pool option with the relevant choices."""
        super().__init__(concurrency.get_available_pool_names())

    def convert(self, value, param, ctx):
        if isinstance(value, type) and issubclass(value, BasePool):
            return value
        value = super().convert(value, param, ctx)
        worker_pool = ctx.obj.app.conf.worker_pool
        if value == 'prefork' and worker_pool:
            value = concurrency.get_implementation(worker_pool)
        else:
            value = concurrency.get_implementation(value)
            if not value:
                value = concurrency.get_implementation(worker_pool)
        return value