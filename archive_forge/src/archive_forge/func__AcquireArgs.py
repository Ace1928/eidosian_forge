from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import re
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import display
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import text
import six
def _AcquireArgs(self):
    """Calls the functions to register the arguments for this module."""
    self._common_type._Flags(self.ai)
    self._common_type.Args(self.ai)
    if self._parent_group:
        for arg in self._parent_group.ai.arguments:
            self.ai.arguments.append(arg)
        if self._parent_group.ai.concept_handler:
            if not self.ai.concept_handler:
                self.ai.add_concepts(handlers.RuntimeHandler())
            for concept_details in self._parent_group.ai.concept_handler._all_concepts:
                try:
                    self.ai.concept_handler.AddConcept(**concept_details)
                except handlers.RepeatedConceptName:
                    raise parser_errors.ArgumentException('repeated concept in {command}: {concept_name}'.format(command=self.dotted_name, concept_name=concept_details['name']))
        for flag in self._parent_group.GetAllAvailableFlags():
            if flag.is_replicated:
                continue
            if flag.do_not_propagate:
                continue
            if flag.is_required:
                continue
            try:
                self.ai.AddFlagActionFromAncestors(flag)
            except argparse.ArgumentError:
                raise parser_errors.ArgumentException('repeated flag in {command}: {flag}'.format(command=self.dotted_name, flag=flag.option_strings))
        self.ai.display_info.AddLowerDisplayInfo(self._parent_group.ai.display_info)