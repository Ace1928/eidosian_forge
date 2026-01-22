import collections
import fnmatch
import glob
import itertools
import os.path
import re
import weakref
from oslo_config import cfg
from oslo_log import log
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.engine import support
class GlobResourceInfo(MapResourceInfo):
    """Store the mapping (with wild cards) of one resource type to another.

    like: OS::Networking::* -> OS::Neutron::*

    Also supports many-to-one mapping (mostly useful together with special
    "OS::Heat::None" resource)

    like: OS::* -> OS::Heat::None
    """
    description = 'Wildcard Mapping'
    __slots__ = tuple()

    def get_resource_info(self, resource_type=None, resource_name=None):
        orig_prefix = self.name[:-1]
        if self.value.endswith('*'):
            new_type = self.value[:-1] + resource_type[len(orig_prefix):]
        else:
            new_type = self.value
        return self.registry.get_resource_info(new_type, resource_name)

    def matches(self, resource_type):
        match = resource_type != self.value and resource_type.startswith(self.name[:-1])
        return match