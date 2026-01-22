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
class NamespaceFlagGroup(BinaryCommandFlag):
    """Encapsulates logic for handling the mutually-exclusive flags --namespace and --all-namespaces."""

    def AddToParser(self, parser):
        mutex_group = parser.add_mutually_exclusive_group()
        NamespaceFlag().AddToParser(mutex_group)
        mutex_group.add_argument('--all-namespaces', default=False, action='store_true', help='List the requested object(s) across all namespaces.')

    def FormatFlags(self, args):
        if args.IsSpecified('all_namespaces'):
            return ['--all-namespaces']
        return NamespaceFlag().FormatFlags(args)