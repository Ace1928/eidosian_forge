from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
class StringFlag(BinaryCommandFlag):
    """A flag that takes a string value that is just passed directly through to the binary."""

    def __init__(self, name, **kwargs):
        super(StringFlag, self).__init__()
        self.arg = base.Argument(name, **kwargs)

    def AddToParser(self, parser):
        return self.arg.AddToParser(parser)

    def FormatFlags(self, args):
        dest_name = _GetDestNameForFlag(self.arg.name)
        if args.IsSpecified(dest_name):
            return [self.arg.name, str(getattr(args, dest_name))]
        return []