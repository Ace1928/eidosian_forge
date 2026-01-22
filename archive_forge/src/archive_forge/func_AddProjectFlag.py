from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def AddProjectFlag(parser, help_text):
    parser.add_argument('--project', metavar='PROJECT_ID', help=help_text)