from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.core import exceptions
class InvalidBuildConfigException(exceptions.Error):
    """Build config message is not valid.

  """

    def __init__(self, path, msg):
        msg = 'validating {path} as build config: {msg}'.format(path=path, msg=msg)
        super(InvalidBuildConfigException, self).__init__(msg)