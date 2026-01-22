from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.metastore import util as api_util
from googlecloudsdk.calliope import base
def GetFederation(release_track=base.ReleaseTrack.GA):
    return api_util.GetClientInstance(release_track=release_track).projects_locations_federations