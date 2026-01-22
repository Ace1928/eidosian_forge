from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_projector
import six
def _CreateProgressTracker(patch_job_name):
    """Creates a progress tracker to display patch status synchronously."""
    stages = [progress_tracker.Stage('Generating instance details...', key='pre-summary'), progress_tracker.Stage('Reporting instance details...', key='with-summary')]
    return progress_tracker.StagedProgressTracker(message='Executing patch job [{0}]'.format(patch_job_name), stages=stages)