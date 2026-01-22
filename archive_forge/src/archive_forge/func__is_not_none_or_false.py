import enum
import functools
from google.api_core import grpc_helpers
from google.api_core.gapic_v1 import client_info
from google.api_core.timeout import TimeToDeadlineTimeout
def _is_not_none_or_false(value):
    return value is not None and value is not False