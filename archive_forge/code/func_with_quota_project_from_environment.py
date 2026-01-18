import abc
import os
import six
from google.auth import _helpers, environment_vars
from google.auth import exceptions
def with_quota_project_from_environment(self):
    quota_from_env = os.environ.get(environment_vars.GOOGLE_CLOUD_QUOTA_PROJECT)
    if quota_from_env:
        return self.with_quota_project(quota_from_env)
    return self