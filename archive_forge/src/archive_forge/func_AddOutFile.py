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
def AddOutFile(parser, positional=False, **kwargs):
    parser.add_argument(_ArgOrFlag('out-file', positional), metavar='OUT-FILE-PATH', help='File path to which secret data is written. If this flag is not provided secret data will be written to stdout in UTF-8 format.', **kwargs)