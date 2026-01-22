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
def AddPrivateIpv6GoogleAccessTypeFlag(api_version, parser, hidden=False):
    """Adds --private-ipv6-google-access-type={disabled|outbound-only|bidirectional} flag."""
    messages = apis.GetMessagesModule('container', api_version)
    util.GetPrivateIpv6GoogleAccessTypeMapper(messages, hidden).choice_arg.AddToParser(parser)