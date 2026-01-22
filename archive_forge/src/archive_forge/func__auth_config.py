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
@staticmethod
def _auth_config(credentials: EngagementCredentials) -> rpcq.ClientAuthConfig:
    return rpcq.ClientAuthConfig(client_secret_key=credentials.client_secret.encode(), client_public_key=credentials.client_public.encode(), server_public_key=credentials.server_public.encode())