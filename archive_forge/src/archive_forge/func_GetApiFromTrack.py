from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
def GetApiFromTrack(track):
    """Returns api version based on the track."""
    if track == base.ReleaseTrack.BETA:
        return 'v1beta2'
    elif track == base.ReleaseTrack.GA:
        return 'v1'