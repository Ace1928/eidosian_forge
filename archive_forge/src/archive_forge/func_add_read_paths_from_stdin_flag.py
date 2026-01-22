from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def add_read_paths_from_stdin_flag(parser, help_text='Read the list of URLs from stdin.'):
    parser.add_argument('--read-paths-from-stdin', '-I', action='store_true', help=help_text)