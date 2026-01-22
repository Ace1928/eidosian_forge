import io
import json
import os
import six
from google.auth import _default
from google.auth import environment_vars
from google.auth import exceptions
def _get_explicit_environ_credentials(quota_project_id=None):
    """Gets credentials from the GOOGLE_APPLICATION_CREDENTIALS environment
    variable."""
    from google.auth import _cloud_sdk
    cloud_sdk_adc_path = _cloud_sdk.get_application_default_credentials_path()
    explicit_file = os.environ.get(environment_vars.CREDENTIALS)
    if explicit_file is not None and explicit_file == cloud_sdk_adc_path:
        return _get_gcloud_sdk_credentials(quota_project_id=quota_project_id)
    if explicit_file is not None:
        credentials, project_id = load_credentials_from_file(os.environ[environment_vars.CREDENTIALS], quota_project_id=quota_project_id)
        return (credentials, project_id)
    else:
        return (None, None)