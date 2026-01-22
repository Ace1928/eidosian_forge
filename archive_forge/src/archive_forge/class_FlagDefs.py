from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
class FlagDefs(object):
    """Base type for all flag builders."""

    def __init__(self):
        self._operations = set()

    def _AddFlag(self, name, **kwargs):
        self._AddOperation(FlagDef(name, **kwargs))

    def _AddOperation(self, operation):
        self._operations.add(operation)

    def ConfigureParser(self, parser):
        for operation in self._operations:
            operation.ConfigureParser(parser)