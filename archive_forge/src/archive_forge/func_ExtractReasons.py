from googlecloudsdk.api_lib.util import exceptions as gcloud_exception
from googlecloudsdk.core import exceptions as gcloud_core_exceptions
def ExtractReasons(e):
    details = e.payload.type_details['ErrorInfo']
    if details is None:
        return None
    return [d['reason'] for d in details]