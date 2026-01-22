from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import copy
import os
import re
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import util as concepts_util
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
class PromptFallthrough(deps.Fallthrough):
    """Fall through to reading from an interactive prompt."""

    def __init__(self, hint):
        super(PromptFallthrough, self).__init__(function=None, hint=hint)

    @abc.abstractmethod
    def _Prompt(self, parsed_args):
        pass

    def _Call(self, parsed_args):
        if not console_io.CanPrompt():
            return None
        return self._Prompt(parsed_args)