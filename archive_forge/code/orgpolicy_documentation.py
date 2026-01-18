from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding_helper
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.orgpolicy import utils as orgpolicy_utils
from googlecloudsdk.api_lib.policy_intelligence import orgpolicy_simulator
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.policy_intelligence.simulator.orgpolicy import utils
from googlecloudsdk.core import log
Parses arguments for the commands.