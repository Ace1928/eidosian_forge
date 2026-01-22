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
def AddJvmMainMutex(parser):
    """Main class or main jar."""
    main_group = parser.add_mutually_exclusive_group(required=True)
    main_group.add_argument('--class', dest='main_class', help='Class contains the main method of the job. The jar file that contains the class must be in the classpath or specified in `jar_files`.')
    main_group.add_argument('--jar', dest='main_jar', help='URI of the main jar file.')