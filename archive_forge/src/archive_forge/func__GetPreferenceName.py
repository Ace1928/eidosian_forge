from apitools.base.py import list_pager
from googlecloudsdk.api_lib.quotas import message_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import common_args
def _GetPreferenceName(request_parent, preference_id):
    if preference_id is None:
        return None
    return request_parent + '/quotaPreferences/' + preference_id