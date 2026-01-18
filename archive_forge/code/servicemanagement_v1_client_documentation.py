from __future__ import absolute_import
from apitools.base.py import base_api
from samples.servicemanagement_sample.servicemanagement_v1 import servicemanagement_v1_messages as messages
from the newest to the oldest.
DEPRECATED. `SubmitConfigSource` with `validate_only=true` will provide.
config conversion moving forward.

Converts an API specification (e.g. Swagger spec) to an
equivalent `google.api.Service`.

      Args:
        request: (ConvertConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConvertConfigResponse) The response message.
      