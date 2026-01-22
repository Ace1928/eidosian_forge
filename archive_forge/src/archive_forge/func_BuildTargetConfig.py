from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import glob
import os
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import yaml
import googlecloudsdk.core.properties
from googlecloudsdk.core.util import files
import requests
import six
import six.moves.urllib.parse
def BuildTargetConfig(messages, config=None, template=None, composite_type=None, properties=None):
    """Construct a TargetConfig from the provided config file with imports.

  Args:
    messages: Object with v2 API messages.
    config: Path to the yaml config file.
    template: Path to the template config file.
    composite_type: name of the composite type config.
    properties: Dictionary of properties, only used if the full_path is a
        template or composite type.

  Returns:
    TargetConfig containing the contents of the config file and the names and
    contents of any imports.

  Raises:
    ConfigError: if the config file or import files cannot be read from
        the specified locations, or if they are malformed.
  """
    config_object = BuildConfig(config=config, template=template, composite_type=composite_type, properties=properties)
    return messages.TargetConfiguration(config=messages.ConfigFile(content=config_object.GetContent()), imports=CreateImports(messages, config_object))