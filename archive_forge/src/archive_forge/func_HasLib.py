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
def HasLib(parsed, name, version=None):
    """Check if the parsed yaml has specified the given library.

  Args:
    parsed: parsed from yaml to python object
    name: str, Name of the library
    version: str, If specified, also matches against the version of the library.

  Returns:
    True if library with optionally the given version is present.
  """
    libs = parsed.libraries or []
    if version:
        return any((lib.name == name and lib.version == version for lib in libs))
    else:
        return any((lib.name == name for lib in libs))