from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.api_lib.dataflow import job_display
from googlecloudsdk.api_lib.dataflow import step_json
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataflow import job_utils
def _PrettyStep(self, step):
    """Prettify a given step, by only extracting certain pieces of info.

    Args:
      step: The step to prettify.
    Returns:
      A dictionary describing the step.
    """
    return {'id': step['name'], 'user_name': step['properties']['user_name']}