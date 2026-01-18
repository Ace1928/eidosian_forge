from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.interconnects.attachments import (
from googlecloudsdk.command_lib.compute.routers import flags as router_flags
from googlecloudsdk.command_lib.compute.routers import router_utils
from googlecloudsdk.command_lib.compute.vpn_tunnels import (
from googlecloudsdk.core import log
Issues requests necessary to update interfaces of the Router.