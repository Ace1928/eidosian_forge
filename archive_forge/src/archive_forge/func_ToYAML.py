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
def ToYAML(self):
    msg = self.AsDict()
    return yaml.dump(msg)