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
def _register_item(self, path, item):
    name = path[-1]
    registry = self._registry
    for key in path[:-1]:
        if key not in registry:
            registry[key] = {}
        registry = registry[key]
    registry[name] = item