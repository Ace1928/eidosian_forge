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
def _ValidateCommandWithPartials(command_data_list, path):
    """Validates that the command with partials do not have duplicated tracks.

  Args:
    command_data_list: List with data loaded from all YAML partials.
    path: [str], A list of group names that got us down to this command group
      with respect to the CLI itself.  This path should be used for things
      like error reporting when a specific element in the tree needs to be
      referenced.

  Raises:
    CommandLoadFailure: If the command is invalid and should not be loaded.
  """
    release_tracks = set()
    for command_data in command_data_list:
        for release_track in command_data['release_tracks']:
            if release_track in release_tracks:
                raise CommandLoadFailure('.'.join(path), Exception(f'Command with partials cannot have duplicated release tracks. Found multiple [{release_track}s]'))
            else:
                release_tracks.add(release_track)