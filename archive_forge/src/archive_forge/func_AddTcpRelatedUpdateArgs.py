from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.health_checks import exceptions as hc_exceptions
def AddTcpRelatedUpdateArgs(parser):
    """Adds parser arguments for update subcommands related to TCP."""
    _AddPortRelatedUpdateArgs(parser)
    AddProxyHeaderRelatedUpdateArgs(parser)
    _AddTcpRelatedArgsImpl(add_info_about_clearing=True, parser=parser)