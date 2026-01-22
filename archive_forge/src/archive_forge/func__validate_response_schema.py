import json
import os
import subprocess
import sys
import time
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
def _validate_response_schema(self, response):
    if 'version' not in response:
        raise exceptions.MalformedError('The executable response is missing the version field.')
    if response['version'] > EXECUTABLE_SUPPORTED_MAX_VERSION:
        raise exceptions.RefreshError('Executable returned unsupported version {}.'.format(response['version']))
    if 'success' not in response:
        raise exceptions.MalformedError('The executable response is missing the success field.')