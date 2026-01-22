from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
def _get_shape_name(self, model):
    if 'shape_name' in model:
        return model['shape_name']
    else:
        return self._name_generator.new_shape_name(model['type'])