import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
@classmethod
def _get_implicit_credentials(cls):
    """Gets credentials implicitly from the environment.

        Checks environment in order of precedence:
        - Environment variable GOOGLE_APPLICATION_CREDENTIALS pointing to
          a file with stored credentials information.
        - Stored "well known" file associated with `gcloud` command line tool.
        - Google App Engine (production and testing)
        - Google Compute Engine production environment.

        Raises:
            ApplicationDefaultCredentialsError: raised when the credentials
                                                fail to be retrieved.
        """
    environ_checkers = [cls._implicit_credentials_from_files, cls._implicit_credentials_from_gae, cls._implicit_credentials_from_gce]
    for checker in environ_checkers:
        credentials = checker()
        if credentials is not None:
            return credentials
    raise ApplicationDefaultCredentialsError(ADC_HELP_MSG)