from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import collections
import copy
import datetime
import enum
import hashlib
import json
import os
import sqlite3
from google.auth import compute_engine as google_auth_compute_engine
from google.auth import credentials as google_auth_creds
from google.auth import exceptions as google_auth_exceptions
from google.auth import external_account as google_auth_external_account
from google.auth import external_account_authorized_user as google_auth_external_account_authorized_user
from google.auth import impersonated_credentials as google_auth_impersonated
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import devshell as c_devshell
from googlecloudsdk.core.credentials import exceptions as c_exceptions
from googlecloudsdk.core.credentials import introspect as c_introspect
from googlecloudsdk.core.util import files
from oauth2client import client
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
import six
def GetQuotaProject(credentials, force_resource_quota=False):
    """Gets the value to use for the X-Goog-User-Project header.

  Args:
    credentials: The credentials that are going to be used for requests.
    force_resource_quota: bool, If true, resource project quota will be used
      even if gcloud is set to use legacy mode for quota. This should be set
      when calling newer APIs that would not work without resource quota.

  Returns:
    str, The project id to send in the header or None to not populate the
    header.
  """
    if credentials is None:
        return None
    quota_project = properties.VALUES.billing.quota_project.Get()
    if _QuotaProjectIsCurrentProject(quota_project):
        if IsUserAccountCredentials(credentials):
            return properties.VALUES.core.project.Get()
        else:
            return None
    elif quota_project == properties.VALUES.billing.LEGACY:
        if force_resource_quota:
            return properties.VALUES.core.project.Get()
        return None
    return quota_project