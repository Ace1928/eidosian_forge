import json
import numbers
from collections import OrderedDict
from functools import update_wrapper
from pprint import pformat
from typing import Any
import click
from click import Context, ParamType
from kombu.utils.objects import cached_property
from celery._state import get_current_app
from celery.signals import user_preload_options
from celery.utils import text
from celery.utils.log import mlevel
from celery.utils.time import maybe_iso8601
class ISO8601DateTime(ParamType):
    """ISO 8601 Date Time argument."""
    name = 'iso-86091'

    def convert(self, value, param, ctx):
        try:
            return maybe_iso8601(value)
        except (TypeError, ValueError) as e:
            self.fail(e)