from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, cast, Tuple, Union, List, Any
from attr import field
import rpcq
from dateutil.parser import parse as parsedate
from dateutil.tz import tzutc
from qcs_api_client.models import EngagementWithCredentials, EngagementCredentials
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from pyquil.api import EngagementManager
from pyquil._version import DOCS_URL
@dataclass
class GetBuffersRequest:
    """
    Request for getting job buffers.
    """
    job_id: str
    'Job for which to get buffers.'
    wait: bool
    'Whether or not to wait until buffers become available.'