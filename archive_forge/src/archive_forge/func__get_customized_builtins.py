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
def _get_customized_builtins(self, operation_model, call_args, request_context):
    service_id = self._service_model.service_id.hyphenize()
    customized_builtins = copy.copy(self._builtins)
    self._event_emitter.emit('before-endpoint-resolution.%s' % service_id, builtins=customized_builtins, model=operation_model, params=call_args, context=request_context)
    return customized_builtins