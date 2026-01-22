from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import importlib
import os
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_release_tracks
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import pkg_resources
from ruamel import yaml
import six
def _GetModuleFromPath(impl_file, path, construction_id):
    """Import the module and dig into it to return the namespace we are after.

  Import the module relative to the top level directory.  Then return the
  actual module corresponding to the last bit of the path.

  Args:
    impl_file: str, The path to the file this was loaded from (for error
      reporting).
    path: [str], A list of group names that got us down to this command group
      with respect to the CLI itself.  This path should be used for things
      like error reporting when a specific element in the tree needs to be
      referenced.
    construction_id: str, A unique identifier for the CLILoader that is
      being constructed.

  Returns:
    The imported module.
  """
    name_to_give = '__calliope__command__.{construction_id}.{name}'.format(construction_id=construction_id, name='.'.join(path).replace('-', '_'))
    try:
        return pkg_resources.GetModuleFromPath(name_to_give, impl_file)
    except Exception as e:
        exceptions.reraise(CommandLoadFailure('.'.join(path), e))