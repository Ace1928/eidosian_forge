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
class EngagementCacheKey(NamedTuple):
    quantum_processor_id: str
    endpoint_id: Optional[str]