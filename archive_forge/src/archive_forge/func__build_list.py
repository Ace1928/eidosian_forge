from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
def _build_list(self, model, shapes):
    member_shape_name = self._get_shape_name(model)
    shape = self._build_initial_shape(model)
    shape['member'] = {'shape': member_shape_name}
    self._build_model(model['member'], shapes, member_shape_name)
    return shape