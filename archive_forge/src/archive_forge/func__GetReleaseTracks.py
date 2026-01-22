from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os.path
from googlecloudsdk.core import branding
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import name_parsing
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def _GetReleaseTracks(release_tracks):
    """Returns a string representation of release tracks.

  Args:
    release_tracks: API versions to generate release tracks for.
  """
    release_tracks_normalized = '[{}]'.format(', '.join([track.upper() for track in sorted(release_tracks)]))
    return release_tracks_normalized