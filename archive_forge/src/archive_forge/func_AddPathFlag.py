from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core.util import files
def AddPathFlag(parser, required=False):
    parser.add_argument('--path', required=required, type=files.ExpandHomeAndVars, default='-', help='Path of the directory or file to output configuration(s). To output configurations to stdout, specify "--path=-".')