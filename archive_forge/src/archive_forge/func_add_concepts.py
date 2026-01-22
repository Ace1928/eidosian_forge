from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import display_info
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core.cache import completion_cache
def add_concepts(self, handler):
    from googlecloudsdk.command_lib.concepts import concept_managers
    if isinstance(handler, concept_managers.RuntimeParser):
        self.data.concepts = handler
        return
    if self.data.concept_handler:
        raise AttributeError('It is not permitted to add two runtime handlers to a command class.')
    self.data.concept_handler = handler