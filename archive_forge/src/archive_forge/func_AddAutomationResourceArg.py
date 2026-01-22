from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddAutomationResourceArg(parser, help_text=None, positional=False, required=True):
    """Add --automation resource argument to the parser.

  Args:
    parser: argparse.ArgumentParser, the parser for the command.
    help_text: help text for this flag.
    positional: if it is a positional flag.
    required: if it is required.
  """
    help_text = help_text or 'The name of the Automation.'
    concept_parsers.ConceptParser.ForResource('automation' if positional else '--automation', GetAutomationResourceSpec(), help_text, required=required, plural=False).AddToParser(parser)