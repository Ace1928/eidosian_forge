import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
@classmethod
def _raise_many(cls, messages):
    if not messages:
        return
    msg = messages.pop(0)
    e = RuntimeError(msg)
    try:
        cls._raise_many(messages)
        raise e
    except RuntimeError as e1:
        raise e from e1