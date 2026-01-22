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
def _GetNewConfigDirectory():
    docker_config = encoding.GetEncodedValue(os.environ, 'DOCKER_CONFIG')
    if docker_config is not None:
        return docker_config
    else:
        return os.path.join(_GetUserHomeDir(), '.docker')