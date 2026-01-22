from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core.util import files
def AddResolveResourcesArg(parser):
    parser.add_argument('--resolve-references', action='store_true', default=False, hidden=True, help='If True, any resource references in the target file PATH will be resolved, and those external resources will be applied as well.')