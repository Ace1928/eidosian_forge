from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import json
import os
import subprocess
import sys
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import semver
import six
from six.moves import urllib
def ReadConfigurationFile(path):
    """Retrieve the full contents of the Docker configuration file.

  Args:
    path: string, path to configuration file

  Returns:
    The full contents of the configuration file as parsed JSON dict.

  Raises:
    ValueError: path is not set.
    InvalidDockerConfigError: config file could not be read as JSON.
  """
    if not path:
        raise ValueError('Docker configuration file path is empty')
    if not os.path.exists(path):
        return {}
    contents = files.ReadFileContents(path)
    if not contents or contents.isspace():
        return {}
    try:
        return json.loads(contents)
    except ValueError as err:
        raise InvalidDockerConfigError('Docker configuration file [{}] could not be read as JSON: {}'.format(path, six.text_type(err)))