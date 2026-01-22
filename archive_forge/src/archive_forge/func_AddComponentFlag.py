from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import json
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def AddComponentFlag(parser):
    """Add optional components flag."""
    help_text = '      List of optional components to be installed on cluster machines.\n\n      The following page documents the optional components that can be\n      installed:\n      https://cloud.google.com/dataproc/docs/concepts/configuring-clusters/optional-components.\n      '
    parser.add_argument('--optional-components', metavar='COMPONENT', type=arg_parsers.ArgList(element_type=lambda val: val.upper()), dest='components', help=help_text)