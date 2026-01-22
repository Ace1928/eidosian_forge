from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddAllowedPscProjects(parser, hidden=False):
    parser.add_argument('--allowed-psc-projects', type=arg_parsers.ArgList(min_length=1), required=False, metavar='PROJECT', help='A comma-separated list of projects. Each project in this list might be represented by a project number (numeric) or by a project ID (alphanumeric). This allows Private Service Connect connections to be established from specified consumer projects.', hidden=hidden)