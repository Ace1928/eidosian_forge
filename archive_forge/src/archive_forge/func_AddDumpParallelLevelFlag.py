from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
def AddDumpParallelLevelFlag(parser):
    """Adds a --dump-parallel-level flag to the given parser."""
    help_text = 'Parallelization level during initial dump of the migration job. If not specified, will be defaulted to OPTIMAL.'
    choices = ['MIN', 'OPTIMAL', 'MAX']
    parser.add_argument('--dump-parallel-level', help=help_text, choices=choices)