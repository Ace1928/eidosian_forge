from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.beyondcorp.app import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.beyondcorp.app import util as command_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def GetVersionedConnectorReq(args, req):
    if args.calliope_command.ReleaseTrack() == base.ReleaseTrack.ALPHA:
        return req.googleCloudBeyondcorpAppconnectorsV1alphaAppConnector
    return req.googleCloudBeyondcorpAppconnectorsV1AppConnector