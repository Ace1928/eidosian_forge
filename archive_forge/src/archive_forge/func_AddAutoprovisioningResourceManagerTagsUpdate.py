from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddAutoprovisioningResourceManagerTagsUpdate(parser):
    """Adds a --autoprovisioning-resource-manager-tags to the given parser on update."""
    help_text = "For an Autopilot cluster, the specified comma-separated resource manager tags\nthat has the GCP_FIREWALL purpose replace the existing tags on all nodes in\nthe cluster.\n\nFor a Standard cluster, the specified comma-separated resource manager tags\nthat has the GCE_FIREWALL purpose are applied to all nodes in the new\nnewly created auto-provisioned node pools. Existing auto-provisioned node pools\nretain the tags that they had before the update. To update tags on an existing\nauto-provisioned node pool, use the node pool level flag\n'--resource-manager-tags'.\n\nExamples:\n\n  $ {command} example-cluster --autoprovisioning-resource-manager-tags=tagKeys/1234=tagValues/2345\n  $ {command} example-cluster --autoprovisioning-resource-manager-tags=my-project/key1=value1\n  $ {command} example-cluster --autoprovisioning-resource-manager-tags=12345/key1=value1,23456/key2=value2\n  $ {command} example-cluster --autoprovisioning-resource-manager-tags=\n\nAll nodes in an Autopilot cluster or all newly created auto-provisioned nodes\nin a Standard cluster, including nodes that are resized or re-created, will have\nthe specified tags on the corresponding Instance object in the Compute Engine API.\nYou can reference these tags in network firewall policy rules. For instructions,\nsee https://cloud.google.com/firewall/docs/use-tags-for-firewalls.\n"
    AddAutoprovisioningResourceManagerTagsFlag(parser, help_text)