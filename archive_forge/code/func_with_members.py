from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
def with_members(self, members):
    """

        :type members: dict
        :param members: The denormalized members.

        :return: self

        """
    self._members = members
    return self