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
@instance_cache
def _get_client_context_params(self):
    """Mapping of param names to client configuration variable"""
    return {param.name: xform_name(param.name) for param in self._service_model.client_context_parameters}