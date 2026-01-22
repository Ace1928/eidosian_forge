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
def _ExtractReleaseTrackImplementation(impl_file, expected_track, implementations):
    """Validates and extracts the correct implementation of the command or group.

  Args:
    impl_file: str, The path to the file this was loaded from (for error
      reporting).
    expected_track: base.ReleaseTrack, The release track we are trying to load.
    implementations: [(func->base._Common, [base.ReleaseTrack])], A list of
    tuples where each item in this list represents a command implementation. The
    first element is a function that returns the implementation, and the second
    element is a list of release tracks it is valid for.

  Raises:
    LayoutException: If there is not exactly one type inheriting
        CommonBase.
    ReleaseTrackNotImplementedException: If there is no command or group
      implementation for the request release track.

  Returns:
    object, The single implementation that matches the expected release track.
  """
    if len(implementations) == 1:
        impl, valid_tracks = implementations[0]
        if not valid_tracks or expected_track in valid_tracks:
            return impl
        raise ReleaseTrackNotImplementedException('No implementation for release track [{0}] for element: [{1}]'.format(expected_track.id, impl_file))
    implemented_release_tracks = set()
    for impl, valid_tracks in implementations:
        if not valid_tracks:
            raise LayoutException('Multiple implementations defined for element: [{0}]. Each must explicitly declare valid release tracks.'.format(impl_file))
        duplicates = implemented_release_tracks & valid_tracks
        if duplicates:
            raise LayoutException('Multiple definitions for release tracks [{0}] for element: [{1}]'.format(', '.join([six.text_type(d) for d in duplicates]), impl_file))
        implemented_release_tracks |= valid_tracks
    valid_commands_or_groups = [impl for impl, valid_tracks in implementations if expected_track in valid_tracks]
    if len(valid_commands_or_groups) != 1:
        raise ReleaseTrackNotImplementedException('No implementation for release track [{0}] for element: [{1}]'.format(expected_track.id, impl_file))
    return valid_commands_or_groups[0]