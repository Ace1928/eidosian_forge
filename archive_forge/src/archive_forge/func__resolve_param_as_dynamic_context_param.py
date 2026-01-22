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
def _resolve_param_as_dynamic_context_param(self, param_name, operation_model, call_args):
    dynamic_ctx_params = self._get_dynamic_context_params(operation_model)
    if param_name in dynamic_ctx_params:
        member_name = dynamic_ctx_params[param_name]
        return call_args.get(member_name)