from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddUseWithNotebook(parser):
    return parser.add_argument('--use-with-notebook', action='store_true', required=False, default=False, help='      Allow Compute Engine VM to be recognized by Cloud AI Notebooks. This\n      automatically sets the content of the flag --use-dl-images flag to be\n      true.\n      ')