from the current environment without the need to copy, save and manage
import abc
import copy
import datetime
import io
import json
import re
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import metrics
from google.oauth2 import sts
from google.oauth2 import utils
@_helpers.copy_docstring(credentials.CredentialsWithQuotaProject)
def with_quota_project(self, quota_project_id):
    kwargs = self._constructor_args()
    kwargs.update(quota_project_id=quota_project_id)
    new_cred = self.__class__(**kwargs)
    new_cred._metrics_options = self._metrics_options
    return new_cred