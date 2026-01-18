from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.recommender import insight
from googlecloudsdk.command_lib.projects import util as project_util
Returns a risk assesment message for IAM policy binding deletion.

  Args:
    release_track: Release track of the recommender.
    project_id: String project ID.
    member: IAM policy binding member.
    member_role: IAM policy binding member role.

  Returns:
    String Active Assist risk warning message to be displayed in IAM policy
    binding deletion prompt.
    If no risk exists, then returns 'None'.
  