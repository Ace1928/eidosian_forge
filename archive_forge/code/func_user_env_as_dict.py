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
def user_env_as_dict(self):
    """Get the environment as a dict, only user-allowed keys."""
    return {env_fmt.RESOURCE_REGISTRY: self.registry.as_dict(), env_fmt.PARAMETERS: self.params, env_fmt.PARAMETER_DEFAULTS: self.param_defaults, env_fmt.EVENT_SINKS: self._event_sinks}