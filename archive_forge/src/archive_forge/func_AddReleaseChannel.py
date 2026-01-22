from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddReleaseChannel(parser):
    parser.add_argument('--release-channel', default='RELEASE_CHANNEL_UNSPECIFIED', help="\n      Release channel a cluster is subscribed to. It supports two values,\n      NONE and REGULAR. NONE is used to opt out of any release channel. Clusters\n      subscribed to the REGULAR channel will be automatically upgraded to\n      versions that are considered GA quality, and cannot be manually upgraded.\n      Additionally, if the REGULAR channel is used, a specific target version\n      cannot be set with the 'version' flag. If left unspecified, the release\n      channel will default to REGULAR.\n      ")