from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
class DuplicateCredentialArgs(GoogleAPIError):
    """Raised when multiple credentials are passed."""
    pass