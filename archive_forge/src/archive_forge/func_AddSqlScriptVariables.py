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
def AddSqlScriptVariables(parser):
    """Add --params flag."""
    parser.add_argument('--vars', type=arg_parsers.ArgDict(), metavar='NAME=VALUE', help='Mapping of query variable names to values (equivalent to the Spark SQL command: SET name="value";).')