from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
def AddOutputFormat(parser, release_track):
    parser.display_info.AddFormat(_RELEASE_TRACK_TO_LIST_FORMAT[release_track])
    parser.display_info.AddTransforms({'requestedRunDuration': _TransformRequestedRunDuration})