import threading
from datetime import datetime
from typing import Dict, NamedTuple, Optional, TYPE_CHECKING
from dateutil.parser import parse as parsedate
from dateutil.tz import tzutc
from qcs_api_client.client import QCSClientConfiguration
from qcs_api_client.models import EngagementWithCredentials, CreateEngagementRequest
from qcs_api_client.operations.sync import create_engagement
from qcs_api_client.types import UNSET
from qcs_api_client.util.errors import QCSHTTPStatusError
from pyquil.api._qcs_client import qcs_client
class QPUUnavailableError(Exception):
    """
    Exception raised when a QPU is unavailable.
    """
    retry_after: Optional[int]
    'The number of seconds after which to retry the engagement request.'

    def __init__(self, retry_after: Optional[str]) -> None:
        if retry_after is not None:
            super().__init__(f'QPU unavailable. Please retry after {retry_after}s.')
        else:
            super().__init__('QPU unavailable.')