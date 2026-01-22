from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.api_lib.recommender import flag_utils
def CreateClient(release_track):
    """Creates Client.

  Args:
    release_track: release_track value, can be ALPHA, BETA, GA

  Returns:
    The versioned client.
  """
    api_version = flag_utils.GetApiVersion(release_track)
    return Recommendation(api_version)