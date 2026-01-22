from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.recommender import insight
from googlecloudsdk.command_lib.projects import util as project_util
def _GetInsightLink(gcloud_insight):
    """Returns a message with a link to the associated recommendation.

  Args:
    gcloud_insight: Insight object returned by the recommender API.

  Returns:
    A string message with a link to the associated recommendation.
  """
    return 'View the full risk assessment at: {0}/view-link/{1}'.format(_RECOMMENDATIONS_HOME_URL, gcloud_insight.name)