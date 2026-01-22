from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def ProcessExternalLoadBalancerAddressPoolsConfig(args, req):
    """Processes the cluster.externalLoadBalancerAddressPools.

  Args:
    args: command line arguments.
    req: API request to be issued
  """
    release_track = args.calliope_command.ReleaseTrack()
    msgs = util.GetMessagesModule(release_track)
    lbdata = args.external_lb_address_pools
    if not lbdata:
        return
    pools = lbdata.get(GDCE_EXTERNAL_LB_CONFIG)
    if not pools:
        return
    err = CheckAddressPoolNameUniqueness(pools)
    if err:
        raise exceptions.InvalidArgumentException('--external-lb-address-pools', f'Duplicate address pool found: {err}')
    mpools = []
    try:
        for pool in pools:
            mpool = messages_util.DictToMessageWithErrorCheck(pool, msgs.ExternalLoadBalancerPool)
            mpools.append(mpool)
    except (messages_util.DecodeError, AttributeError, KeyError) as err:
        raise exceptions.InvalidArgumentException('--external-lb-address-pools', "'{}'".format(err.args[0] if err.args else err))
    if mpools:
        req.cluster.externalLoadBalancerAddressPools = mpools