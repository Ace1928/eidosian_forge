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
def _ImplementationsFromYaml(path, data, yaml_command_translator):
    """Gets all the release track command implementations from the yaml file.

  Args:
    path: [str], A list of group names that got us down to this command group
      with respect to the CLI itself.  This path should be used for things
      like error reporting when a specific element in the tree needs to be
      referenced.
    data: dict, The loaded yaml data.
    yaml_command_translator: YamlCommandTranslator, An instance of a translator
      to use to load the yaml data.

  Raises:
    CommandLoadFailure: If the command is invalid and cannot be loaded.

  Returns:
    [(func->base._Common, [base.ReleaseTrack])], A list of tuples that can be
    passed to _ExtractReleaseTrackImplementation. Each item in this list
    represents a command implementation. The first element is a function that
    returns the implementation, and the second element is a list of release
    tracks it is valid for.
  """
    if not yaml_command_translator:
        raise CommandLoadFailure('.'.join(path), Exception('No yaml command translator has been registered'))
    implementations = [(lambda i=i: yaml_command_translator.Translate(path, i), {base.ReleaseTrack.FromId(t) for t in i.get('release_tracks', [])}) for i in command_release_tracks.SeparateDeclarativeCommandTracks(data)]
    return implementations