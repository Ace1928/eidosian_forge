from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
class NoGRPCInstalledError(exceptions.ToolException):
    """Unable to import grpc-based modules."""

    def __init__(self):
        super(NoGRPCInstalledError, self).__init__(_GrpcSetupHelpMessage())