from __future__ import absolute_import
from enum import Enum
from google.api_core.exceptions import GoogleAPICallError
from typing import Optional
class AcknowledgeStatus(Enum):
    SUCCESS = 1
    PERMISSION_DENIED = 2
    FAILED_PRECONDITION = 3
    INVALID_ACK_ID = 4
    OTHER = 5