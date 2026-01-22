from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import api_util as fleet_api_util
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.core import exceptions
import six
def _GetApiVersionFromReleaseTrack(release_track):
    if release_track == base.ReleaseTrack.ALPHA:
        return 'v1alpha'
    raise ValueError('Invalid release track: ' + release_track)