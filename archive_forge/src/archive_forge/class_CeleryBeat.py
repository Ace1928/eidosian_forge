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
class CeleryBeat(ParamType):
    """Celery Beat flag."""
    name = 'beat'

    def convert(self, value, param, ctx):
        if ctx.obj.app.IS_WINDOWS and value:
            self.fail('-B option does not work on Windows.  Please run celery beat as a separate service.')
        return value