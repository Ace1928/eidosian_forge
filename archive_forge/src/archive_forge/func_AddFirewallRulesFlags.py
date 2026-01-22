from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.api_lib.app import logs_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.docker import docker
from googlecloudsdk.third_party.appengine.api import appinfo
def AddFirewallRulesFlags(parser, required):
    """Add the common flags to a firewall-rules command."""
    parser.add_argument('--source-range', required=required, help='An IP address or range in CIDR notation or the ```*``` wildcard to match all traffic.')
    parser.add_argument('--action', required=required, choices=['ALLOW', 'DENY'], type=lambda x: x.upper(), help='Allow or deny matched traffic.')
    parser.add_argument('--description', help='A text description of the rule.')