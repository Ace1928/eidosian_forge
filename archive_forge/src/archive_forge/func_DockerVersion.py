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
def DockerVersion(self):
    if not self._version:
        version_str = six.text_type(client_utils.GetDockerVersion())
        self._version = semver.LooseVersion(version_str)
    return self._version