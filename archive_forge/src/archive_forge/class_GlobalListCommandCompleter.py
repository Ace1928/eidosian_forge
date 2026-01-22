from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.resource_manager import completers as resource_manager_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
class GlobalListCommandCompleter(ListCommandCompleter):
    """A global resource list command completer."""

    def ParameterInfo(self, parsed_args, argument):
        return ListCommandParameterInfo(parsed_args, argument, self.collection, additional_params=['global'], updaters=COMPLETERS_BY_CONVENTION)