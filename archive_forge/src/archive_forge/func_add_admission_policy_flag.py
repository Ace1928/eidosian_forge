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
def add_admission_policy_flag(parser):
    parser.add_argument('--admission-policy', choices=['ADMIT_ON_FIRST_MISS', 'ADMIT_ON_SECOND_MISS'], help='The cache admission policy decides for each cache miss, whether to insert the missed block or not.')