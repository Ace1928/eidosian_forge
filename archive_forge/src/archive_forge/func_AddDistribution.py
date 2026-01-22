from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddDistribution(parser, required=False):
    help_text = '\nSet the base platform type of the cluster to attach.\n\nExamples:\n\n  $ {command} --distribution=aks\n  $ {command} --distribution=eks\n'
    parser.add_argument('--distribution', required=required, help=help_text)