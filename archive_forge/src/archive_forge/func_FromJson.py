from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.core.docker import client_lib as client_utils
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import semver
import six
@classmethod
def FromJson(cls, json_string, path):
    """Build a Configuration object from a JSON string.

    Args:
      json_string: string, json content for Configuration
      path: string, file path to Docker Configuation File

    Returns:
      a Configuration object
    """
    if not json_string or json_string.isspace():
        config_dict = {}
    else:
        config_dict = json.loads(json_string)
    return Configuration(config_dict, path)