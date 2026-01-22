import asyncio
import logging
import sys
import tempfile
import traceback
from contextlib import aclosing
from asgiref.sync import ThreadSensitiveContext, sync_to_async
from django.conf import settings
from django.core import signals
from django.core.exceptions import RequestAborted, RequestDataTooBig
from django.core.handlers import base
from django.http import (
from django.urls import set_script_prefix
from django.utils.functional import cached_property
def _get_scheme(self):
    return self.scope.get('scheme') or super()._get_scheme()