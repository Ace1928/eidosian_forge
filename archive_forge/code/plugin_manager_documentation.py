import collections
import itertools
import sys
from oslo_config import cfg
from oslo_log import log
from heat.common import plugin_loader
Iterate over the mappings from all modules in the plugin manager.

        Mappings are returned as a list of (key, value) tuples.
        