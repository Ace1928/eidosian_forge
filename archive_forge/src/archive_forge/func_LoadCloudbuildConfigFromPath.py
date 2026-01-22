from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.core import exceptions
def LoadCloudbuildConfigFromPath(path, messages, params=None):
    """Load a cloudbuild config file into a Build message.

  Args:
    path: str. Path to the JSON or YAML data to be decoded.
    messages: module, The messages module that has a Build type.
    params: dict, parameters to substitute into a templated Build spec.

  Raises:
    files.MissingFileError: If the file does not exist.
    ParserError: If there was a problem parsing the file as a dict.
    ParseProtoException: If there was a problem interpreting the file as the
      given message type.
    InvalidBuildConfigException: If the build config has illegal values.

  Returns:
    Build message, The build that got decoded.
  """
    build = cloudbuild_util.LoadMessageFromPath(path, messages.Build, _BUILD_CONFIG_FRIENDLY_NAME, _SKIP_CAMEL_CASE)
    build = FinalizeCloudbuildConfig(build, path, params)
    return build