from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
class MapShape(Shape):

    @CachedProperty
    def key(self):
        return self._resolve_shape_ref(self._shape_model['key'])

    @CachedProperty
    def value(self):
        return self._resolve_shape_ref(self._shape_model['value'])