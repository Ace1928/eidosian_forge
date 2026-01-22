from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import appinfo_includes
from googlecloudsdk.third_party.appengine.api import croninfo
from googlecloudsdk.third_party.appengine.api import dispatchinfo
from googlecloudsdk.third_party.appengine.api import queueinfo
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.datastore import datastore_index
def _CheckIllegalAttribute(name, yaml_info, extractor_func, file_path, msg=''):
    """Validates that an illegal attribute is not set.

  Args:
    name: str, The name of the attribute in the yaml files.
    yaml_info: AppInfoExternal, The yaml to validate.
    extractor_func: func(AppInfoExternal)->str, A function to extract the
      value of the attribute from a _YamlInfo object.
    file_path: str, The path of file from which yaml_info was parsed.
    msg: str, Message to couple with the error

  Raises:
      YamlValidationError: If illegal attribute is set.

  """
    attribute = extractor_func(yaml_info)
    if attribute is not None:
        raise YamlValidationError('The [{0}] field is specified in file [{1}]. '.format(name, file_path) + msg)