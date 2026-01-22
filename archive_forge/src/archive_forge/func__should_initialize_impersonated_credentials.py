from the current environment without the need to copy, save and manage
import abc
import copy
from dataclasses import dataclass
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
def _should_initialize_impersonated_credentials(self):
    return self._service_account_impersonation_url is not None and self._impersonated_credentials is None