import abc
from oslo_utils import excutils
from taskflow import logging
from taskflow import states
from taskflow.types import failure
from taskflow.types import notifier
def _bulk_deregister(notifier, registered, details_filter=None):
    """Bulk deregisters callbacks associated with many states."""
    while registered:
        state, cb = registered.pop()
        notifier.deregister(state, cb, details_filter=details_filter)