from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
def _resolve_shape_ref(self, shape_ref):
    return self._shape_resolver.resolve_shape_ref(shape_ref)