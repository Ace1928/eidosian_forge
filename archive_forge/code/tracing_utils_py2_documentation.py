from functools import wraps
import sentry_sdk
from sentry_sdk import get_current_span
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.utils import logger, qualname_from_function

    Decorator to add child spans for functions.

    This is the Python 2 compatible version of the decorator.
    Duplicated code from ``sentry_sdk.tracing_utils_python3.start_child_span_decorator``.

    See also ``sentry_sdk.tracing.trace()``.
    