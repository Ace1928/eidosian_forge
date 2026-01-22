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
def AddAutoprovisioningNetworkTagsCreate(parser):
    AddAutoprovisioningNetworkTagsFlag(parser, 'Applies the given Compute Engine tags (comma separated) on all nodes in the auto-provisioned node pools of the new Standard cluster or the new Autopilot cluster.\n\nExamples:\n\n  $ {command} example-cluster --autoprovisioning-network-tags=tag1,tag2\n\nNew nodes in auto-provisioned node pools, including ones created by resize or recreate, will have these tags\non the Compute Engine API instance object and can be used in firewall rules.\nSee https://cloud.google.com/sdk/gcloud/reference/compute/firewall-rules/create\nfor examples.\n')