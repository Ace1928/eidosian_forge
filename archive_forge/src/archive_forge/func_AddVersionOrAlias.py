from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.secrets import completers as secrets_completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import resources
def AddVersionOrAlias(parser, purpose, positional=False, **kwargs):
    concept_parsers.ConceptParser.ForResource(name=_ArgOrFlag('version', positional), resource_spec=GetVersionResourceSpec(), group_help="Numeric secret version {} or a configured alias (including 'latest' to use the latest version).".format(purpose), **kwargs).AddToParser(parser)