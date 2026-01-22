import importlib.metadata
import logging
import re
import sys
import yaml
from oslo_config import cfg
from oslo_config import generator
def _validate_opt(group, option, opt_data):
    if group not in opt_data['options']:
        return False
    name_data = [o['name'] for o in opt_data['options'][group]['opts']]
    name_data += [o.get('dest') for o in opt_data['options'][group]['opts']]
    return option in name_data