from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import textwrap
from typing import Mapping
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def GetVpcNetwork(record):
    """Returns the VPC Network setting.

  Either the values of the vpc-access-connector and vpc-access-egress, or the
  values of the network and subnetwork in network-interfaces annotation and
  vpc-access-egress.

  Args:
    record: A dictionary-like object containing the VPC_ACCESS_ANNOTATION and
      EGRESS_SETTINGS_ANNOTATION keys.
  """
    connector = record.get(container_resource.VPC_ACCESS_ANNOTATION, '')
    if connector:
        return cp.Labeled([('Connector', connector), ('Egress', record.get(container_resource.EGRESS_SETTINGS_ANNOTATION, ''))])
    original_value = record.get(k8s_object.NETWORK_INTERFACES_ANNOTATION, '')
    if not original_value:
        return ''
    try:
        network_interface = json.loads(original_value)[0]
        return cp.Labeled([('Network', network_interface.get('network', '')), ('Subnet', network_interface.get('subnetwork', '')), ('Egress', record.get(container_resource.EGRESS_SETTINGS_ANNOTATION, ''))])
    except Exception:
        return ''