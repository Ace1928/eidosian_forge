from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.accesscontextmanager import zones as zones_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import perimeters
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.args import repeated
def _GetIngressEgressFieldValue(args, field_name, base_config_value, has_spec):
    """Returns the ingress/egress field value to use for the update operation."""
    ingress_egress_field = perimeters.ParseUpdateDirectionalPoliciesArgs(args, field_name)
    if not has_spec and ingress_egress_field is None:
        ingress_egress_field = base_config_value
    return ingress_egress_field