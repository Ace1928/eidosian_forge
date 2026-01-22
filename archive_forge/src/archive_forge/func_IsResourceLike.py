from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
import ruamel.yaml as ryaml
def IsResourceLike(item):
    """Return True if item is a dict like object or list of dict like objects."""
    return yaml.dict_like(item) or (yaml.list_like(item) and all((yaml.dict_like(x) for x in item)))