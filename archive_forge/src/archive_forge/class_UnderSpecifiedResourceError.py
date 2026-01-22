from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import enum
import functools
import re
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.api_lib.compute.regions import service as regions_service
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import scope_prompter
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.util import text
import six
class UnderSpecifiedResourceError(exceptions.Error):
    """Raised when argument is required additional scope to be resolved."""

    def __init__(self, underspecified_names, flag_names):
        phrases = ('one of ', 'flags') if len(flag_names) > 1 else ('', 'flag')
        super(UnderSpecifiedResourceError, self).__init__('Underspecified resource [{3}]. Specify {0}the [{1}] {2}.'.format(phrases[0], ', '.join(sorted(flag_names)), phrases[1], ', '.join(underspecified_names)))