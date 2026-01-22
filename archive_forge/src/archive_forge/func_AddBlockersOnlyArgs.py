from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import base
def AddBlockersOnlyArgs(parser):
    parser.add_argument('--blockers-only', metavar='BLOCKERS_ONLY', required=False, default=False, help='Determines whether to perform analysis against blockers only. Leaving it empty means the full analysis will be performed including warnings and blockers for the specified resource move.')