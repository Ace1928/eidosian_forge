from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from containerregistry.client import docker_name
from googlecloudsdk.core.exceptions import Error
import six
from six.moves import urllib
class BadImageUrlError(Error):
    """Raised when a container image URL cannot be parsed successfully."""