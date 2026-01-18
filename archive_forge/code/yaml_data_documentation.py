from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import pkg_resources
Constructs a ResourceYAMLData from a standard resource_path.

    Args:
      resource_path: string, the dotted path of the resources.yaml file, e.g.
        iot.device or compute.instance.

    Returns:
      A ResourceYAMLData object.

    Raises:
      InvalidResourcePathError: invalid resource_path string.
    