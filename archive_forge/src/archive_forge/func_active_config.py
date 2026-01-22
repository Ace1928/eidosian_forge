import os
import json
import pyomo.common.envvar as envvar
from pyomo.common.config import ConfigBase, ConfigBlock, ConfigValue, ADVANCED_OPTION
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
import logging
def active_config(self):
    return self._options_stack[-1]