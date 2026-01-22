from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core.util import files
def AddOnErrorFlag(parser):
    parser.add_argument('--on-error', choices=['continue', 'halt', 'ignore'], default='ignore', help='Determines behavior when a recoverable error is encountered while exporting a resource. To stop execution when encountering an error, specify "halt". To log errors when encountered and continue the export, specify "continue". To continue when errors are encountered without logging, specify "ignore".')