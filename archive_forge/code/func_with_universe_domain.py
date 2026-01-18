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
@_helpers.copy_docstring(credentials.CredentialsWithUniverseDomain)
def with_universe_domain(self, universe_domain):
    kwargs = self._constructor_args()
    kwargs.update(universe_domain=universe_domain)
    new_cred = self.__class__(**kwargs)
    new_cred._metrics_options = self._metrics_options
    return new_cred