import os
import random
import six
from six.moves import http_client
import six.moves.urllib.error as urllib_error
import six.moves.urllib.parse as urllib_parse
import six.moves.urllib.request as urllib_request
from apitools.base.protorpclite import messages
from apitools.base.py import encoding_helper as encoding
from apitools.base.py import exceptions
def MapRequestParams(params, request_type):
    """Perform any renames/remappings needed for URL construction.

    Currently, we have several ways to customize JSON encoding, in
    particular of field names and enums. This works fine for JSON
    bodies, but also needs to be applied for path and query parameters
    in the URL.

    This function takes a dictionary from param names to values, and
    performs any registered mappings. We also need the request type (to
    look up the mappings).

    Args:
      params: (dict) Map from param names to values
      request_type: (protorpc.messages.Message) request type for this API call

    Returns:
      A new dict of the same size, with all registered mappings applied.
    """
    new_params = dict(params)
    for param_name, value in params.items():
        field_remapping = encoding.GetCustomJsonFieldMapping(request_type, python_name=param_name)
        if field_remapping is not None:
            new_params[field_remapping] = new_params.pop(param_name)
            param_name = field_remapping
        if isinstance(value, messages.Enum):
            new_params[param_name] = encoding.GetCustomJsonEnumMapping(type(value), python_name=str(value)) or str(value)
    return new_params