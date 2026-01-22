from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def GetSink(self, parent, sink_ref):
    """Returns a sink specified by the arguments."""
    return util.GetClient().projects_sinks.Get(util.GetMessages().LoggingProjectsSinksGetRequest(sinkName=util.CreateResourceName(parent, 'sinks', sink_ref.sinksId)))