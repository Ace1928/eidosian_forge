from __future__ import absolute_import
import sys
from sentry_sdk._compat import reraise
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk import Hub
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.tracing import Transaction, TRANSACTION_SOURCE_TASK
from sentry_sdk.utils import (
def _sentry_create_worker(*args, **kwargs):
    hub = Hub.current
    if hub.get_integration(ArqIntegration) is None:
        return old_create_worker(*args, **kwargs)
    settings_cls = args[0]
    if hasattr(settings_cls, 'functions'):
        settings_cls.functions = [_get_arq_function(func) for func in settings_cls.functions]
    if hasattr(settings_cls, 'cron_jobs'):
        settings_cls.cron_jobs = [_get_arq_cron_job(cron_job) for cron_job in settings_cls.cron_jobs]
    if 'functions' in kwargs:
        kwargs['functions'] = [_get_arq_function(func) for func in kwargs['functions']]
    if 'cron_jobs' in kwargs:
        kwargs['cron_jobs'] = [_get_arq_cron_job(cron_job) for cron_job in kwargs['cron_jobs']]
    return old_create_worker(*args, **kwargs)