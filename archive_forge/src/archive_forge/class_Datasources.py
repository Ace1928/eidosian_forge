from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Datasources(base.Group):
    """Cloud DLP Commands for analyzing Google Cloud data repositories.

  Cloud DLP Commands for inspecting and analyzing sensitive data in Google Cloud
  data repositories.

  See [Inspecting Storage and Databases for Sensitive Data]
  (https://cloud.google.com/dlp/docs/inspecting-storage)
  for more details.
  """