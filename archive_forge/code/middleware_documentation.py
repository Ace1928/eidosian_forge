from django import VERSION as DJANGO_VERSION
from sentry_sdk import Hub
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.utils import (

Create spans from Django middleware invocations
