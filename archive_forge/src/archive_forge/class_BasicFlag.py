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
class BasicFlag(BinaryCommandFlag):
    """Encapsulates a flag that is passed through as-is when present."""

    def __init__(self, name, **kwargs):
        super(BasicFlag, self).__init__()
        self.arg = base.Argument(name, default=False, action='store_true', **kwargs)

    def AddToParser(self, parser):
        return self.arg.AddToParser(parser)

    def FormatFlags(self, args):
        dest_name = _GetDestNameForFlag(self.arg.name)
        if args.IsSpecified(dest_name):
            return [self.arg.name]
        return []