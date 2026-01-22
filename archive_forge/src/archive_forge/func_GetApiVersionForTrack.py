from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
def GetApiVersionForTrack(release_track=base.ReleaseTrack.GA):
    return _VERSION_MAP.get(release_track)