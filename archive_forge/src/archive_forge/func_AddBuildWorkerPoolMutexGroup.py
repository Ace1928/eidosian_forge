from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from argcomplete.completers import DirectoriesCompleter
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.eventarc import flags as eventarc_flags
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
def AddBuildWorkerPoolMutexGroup(parser):
    """Add flag for specifying Build Worker Pool to the parser."""
    mutex_group = parser.add_group(mutex=True)
    mutex_group.add_argument('--build-worker-pool', help='        Name of the Cloud Build Custom Worker Pool that should be used to build\n        the function. The format of this field is\n        `projects/${PROJECT}/locations/${LOCATION}/workerPools/${WORKERPOOL}`\n        where ${PROJECT} is the project id and ${LOCATION} is the location where\n        the worker pool is defined and ${WORKERPOOL} is the short name of the\n        worker pool.\n      ')
    mutex_group.add_argument('--clear-build-worker-pool', action='store_true', help='        Clears the Cloud Build Custom Worker Pool field.\n      ')