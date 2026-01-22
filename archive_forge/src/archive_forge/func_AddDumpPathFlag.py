from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.calliope import arg_parsers
def AddDumpPathFlag(parser):
    """Adds a --dump-path flag to the given parser."""
    help_text = '    Path to the dump file in Google Cloud Storage, in the format:\n    `gs://[BUCKET_NAME]/[OBJECT_NAME]`.\n    '
    parser.add_argument('--dump-path', help=help_text)