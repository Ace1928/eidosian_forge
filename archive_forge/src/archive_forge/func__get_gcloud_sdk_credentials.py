import io
import json
import os
import six
from google.auth import _default
from google.auth import environment_vars
from google.auth import exceptions
def _get_gcloud_sdk_credentials(quota_project_id=None):
    """Gets the credentials and project ID from the Cloud SDK."""
    from google.auth import _cloud_sdk
    credentials_filename = _cloud_sdk.get_application_default_credentials_path()
    if not os.path.isfile(credentials_filename):
        return (None, None)
    credentials, project_id = load_credentials_from_file(credentials_filename, quota_project_id=quota_project_id)
    if not project_id:
        project_id = _cloud_sdk.get_project_id()
    return (credentials, project_id)