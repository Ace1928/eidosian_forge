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
def _create_default_metrics_options(self):
    metrics_options = {}
    if self._service_account_impersonation_url:
        metrics_options['sa-impersonation'] = 'true'
    else:
        metrics_options['sa-impersonation'] = 'false'
    if self._service_account_impersonation_options.get('token_lifetime_seconds'):
        metrics_options['config-lifetime'] = 'true'
    else:
        metrics_options['config-lifetime'] = 'false'
    return metrics_options