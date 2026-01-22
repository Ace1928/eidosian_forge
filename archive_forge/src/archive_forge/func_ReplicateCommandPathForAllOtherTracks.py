from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import os
import re
import sys
import types
import uuid
import argcomplete
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import backend
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import pkg_resources
import six
def ReplicateCommandPathForAllOtherTracks(self, command_path):
    """Finds other release tracks this command could be in.

    The returned values are not necessarily guaranteed to exist because the
    commands could be disabled for that particular release track.  It is up to
    the caller to determine if the commands actually exist before attempting
    use.

    Args:
      command_path: [str], The path of the command being run.

    Returns:
      {ReleaseTrack: [str]}, A mapping of release track to command path of other
      places this command could be found.
    """
    if len(command_path) < 2:
        return []
    track = calliope_base.ReleaseTrack.FromPrefix(command_path[1])
    if track and track not in self.__release_tracks:
        track = None
    root = command_path[0]
    sub_path = command_path[2:] if track else command_path[1:]
    if not sub_path:
        return []
    results = dict()
    for t in self.__release_tracks:
        results[t] = [root] + [t.prefix] + sub_path
    if track:
        del results[track]
        results[calliope_base.ReleaseTrack.GA] = [root] + sub_path
    return results