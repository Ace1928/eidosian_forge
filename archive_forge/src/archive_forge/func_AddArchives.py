from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import json
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def AddArchives(parser):
    parser.add_argument('--archives', type=arg_parsers.ArgList(), metavar='ARCHIVE', default=[], help='Archives to be extracted into the working directory. Supported file types: .jar, .tar, .tar.gz, .tgz, and .zip.')