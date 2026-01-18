import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
Updates the state of an HMAC key.

      Args:
        request: (StorageProjectsHmacKeysUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (HmacKeyMetadata) The response message.
      