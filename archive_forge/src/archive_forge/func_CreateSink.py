from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def CreateSink(self, parent, sink_data, custom_writer_identity):
    """Creates a v2 sink specified by the arguments."""
    messages = util.GetMessages()
    return util.GetClient().projects_sinks.Create(messages.LoggingProjectsSinksCreateRequest(parent=parent, logSink=messages.LogSink(**sink_data), uniqueWriterIdentity=True, customWriterIdentity=custom_writer_identity))