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
def _LoadCommandWithPartials(impl_file, path):
    """Loads all YAML partials for a command with partials based on conventions.

  Partial files are loaded using _CustomLoadYamlFile as normal YAML commands.

  Conventions:
  - Partials should be placed in subfolder `_partials`.
  - File names of partials should match the main command name and follow this
  format: _[command_name]_[version|release_track].yaml
  - Release tracks should not be duplicatd across all partials.

  Args:
    impl_file: file path to the main YAML command implementation.
    path: [str], A list of group names that got us down to this command group
      with respect to the CLI itself.  This path should be used for things
      like error reporting when a specific element in the tree needs to be
      referenced.

  Returns:
    List with data loaded from partial YAML files for the main command.
  """
    file_name = os.path.basename(impl_file)
    command_name = file_name[:-5]
    partials_dir = os.path.join(os.path.dirname(impl_file), PARTIALS_DIR)
    partial_files = pkg_resources.GetFilesFromDirectory(partials_dir, f'_{command_name}_*.yaml')
    command_data_list = []
    for partial_file in partial_files:
        command_data_list.extend(_CustomLoadYamlFile(partial_file))
    _ValidateCommandWithPartials(command_data_list, path)
    return command_data_list