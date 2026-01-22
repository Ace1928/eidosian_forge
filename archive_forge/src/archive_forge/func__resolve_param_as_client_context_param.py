import copy
import logging
import re
from enum import Enum
from botocore import UNSIGNED, xform_name
from botocore.auth import AUTH_TYPE_MAPS, HAS_CRT
from botocore.crt import CRT_SUPPORTED_AUTH_TYPES
from botocore.endpoint_provider import EndpointProvider
from botocore.exceptions import (
from botocore.utils import ensure_boolean, instance_cache
def _resolve_param_as_client_context_param(self, param_name):
    client_ctx_params = self._get_client_context_params()
    if param_name in client_ctx_params:
        client_ctx_varname = client_ctx_params[param_name]
        return self._client_context.get(client_ctx_varname)