from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import xml.etree.cElementTree as element_tree
from googlecloudsdk.command_lib.metastore import parsers
from googlecloudsdk.core import properties
def GenerateNetworkConfigFromSubnetList(unused_ref, args, req):
    """Generates the NetworkConfig message from the list of subnetworks.

  Args:
    args: The request arguments.
    req: A request with `service` field.

  Returns:
    A request with network configuration field if `consumer-subnetworks` is
    present in the arguments.
  """
    if args.consumer_subnetworks:
        req.service.networkConfig = {'consumers': [{'subnetwork': parsers.ParseSubnetwork(s, args.location)} for s in args.consumer_subnetworks]}
    return req