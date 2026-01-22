import abc
from oslo_utils import excutils
from taskflow import logging
from taskflow import states
from taskflow.types import failure
from taskflow.types import notifier
def _retry_matcher(details):
    """Matches retry details emitted."""
    if not details:
        return False
    if 'retry_name' in details and 'retry_uuid' in details:
        return True
    return False