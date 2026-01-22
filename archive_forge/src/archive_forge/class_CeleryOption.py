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
class CeleryOption(click.Option):
    """Customized option for Celery."""

    def get_default(self, ctx, *args, **kwargs):
        if self.default_value_from_context:
            self.default = ctx.obj[self.default_value_from_context]
        return super().get_default(ctx, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        """Initialize a Celery option."""
        self.help_group = kwargs.pop('help_group', None)
        self.default_value_from_context = kwargs.pop('default_value_from_context', None)
        super().__init__(*args, **kwargs)