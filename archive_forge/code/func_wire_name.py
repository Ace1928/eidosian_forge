from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
@property
def wire_name(self):
    """The wire name of the operation.

        In many situations this is the same value as the
        ``name``, value, but in some services, the operation name
        exposed to the user is different from the operation name
        we send across the wire (e.g cloudfront).

        Any serialization code should use ``wire_name``.

        """
    return self._operation_model.get('name')