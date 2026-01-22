import concurrent.futures
import contextvars
import logging
import sys
from types import GenericAlias
from . import base_futures
from . import events
from . import exceptions
from . import format_helpers
def _call_check_cancel(destination):
    if destination.cancelled():
        if source_loop is None or source_loop is dest_loop:
            source.cancel()
        else:
            source_loop.call_soon_threadsafe(source.cancel)