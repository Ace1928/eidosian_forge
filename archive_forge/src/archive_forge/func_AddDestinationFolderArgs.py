from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import base
def AddDestinationFolderArgs(parser):
    parser.add_argument('--destination-folder', metavar='FOLDER_ID', required=False, help='The destination folder ID to perform the analysis.')