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
def _resolve_variant(self, tags, endpoint_data, service_defaults, partition_defaults):
    result = {}
    for variants in [endpoint_data, service_defaults, partition_defaults]:
        variant = self._retrieve_variant_data(variants, tags)
        if variant:
            self._merge_keys(variant, result)
    return result