from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc import flags as scc_flags
from googlecloudsdk.command_lib.scc import util as scc_util
from googlecloudsdk.command_lib.scc.muteconfigs import flags
from googlecloudsdk.command_lib.scc.muteconfigs import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _GenerateMuteConfig(args, req, version='v1'):
    """Updates parent and Generates a mute config."""
    req.parent = util.ValidateAndGetParent(args)
    if req.parent is not None:
        if version == 'v2':
            req.parent = util.ValidateAndGetRegionalizedParent(args, req.parent)
        req.muteConfigId = util.ValidateAndGetMuteConfigId(args)
    else:
        args.location = scc_util.ValidateAndGetLocation(args, version)
        mute_config = util.ValidateAndGetMuteConfigFullResourceName(args, version)
        req.muteConfigId = util.GetMuteConfigIdFromFullResourceName(mute_config)
        req.parent = util.GetParentFromFullResourceName(mute_config, version)
    args.filter = ''
    return req