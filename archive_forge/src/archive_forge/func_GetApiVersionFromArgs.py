from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def GetApiVersionFromArgs(args):
    """Return API version based on args.

  Update this whenever there is a new version.

  Args:
    args: The argparse namespace.

  Returns:
    API version (e.g. v1alpha or v1beta).

  Raises:
    UnsupportedReleaseTrackError: If invalid release track from args.
  """
    release_track = args.calliope_command.ReleaseTrack()
    if release_track == base.ReleaseTrack.ALPHA:
        return 'v1alpha'
    if release_track == base.ReleaseTrack.BETA:
        return 'v1beta'
    if release_track == base.ReleaseTrack.GA:
        return 'v1'
    raise UnsupportedReleaseTrackError(release_track)