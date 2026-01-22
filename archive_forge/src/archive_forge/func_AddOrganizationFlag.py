from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def AddOrganizationFlag(parser, help_text):
    parser.add_argument('--organization', metavar='ORGANIZATION_ID', help=help_text)