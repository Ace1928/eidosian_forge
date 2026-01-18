from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import xpn_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.xpn import util as command_lib_util
from googlecloudsdk.command_lib.organizations import flags as organizations_flags
from googlecloudsdk.core import properties
List shared VPC host projects in a given organization.