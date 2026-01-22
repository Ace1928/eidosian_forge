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
Reads configuration file and meta-data from default Docker location.

    Reads configuration file and meta-data from default Docker location. Returns
    a Configuration object containing the full contents of the configuration
    file, and the configuration file path.

    Args:
      path: string, path to look for the Docker config file. If empty will
        attempt to read from the new config location (default).

    Returns:
      A Configuration object

    Raises:
      ValueError: path or is_new_format are not set.
      InvalidDockerConfigError: config file could not be read as JSON.
    