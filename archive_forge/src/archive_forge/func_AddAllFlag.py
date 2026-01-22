from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core.util import files
def AddAllFlag(parser, collection='collection'):
    parser.add_argument('--all', action='store_true', help='Retrieve all resources within the {}. If `--path` is specified and is a valid directory, resources will be output as individual files based on resource name and scope. If `--path` is not specified, resources will be streamed to stdout.'.format(collection))