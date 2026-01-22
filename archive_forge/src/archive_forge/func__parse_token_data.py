import io
import json
import os
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
def _parse_token_data(self, token_content, format_type='text', subject_token_field_name=None):
    content, filename = token_content
    if format_type == 'text':
        token = content
    else:
        try:
            response_data = json.loads(content)
            token = response_data[subject_token_field_name]
        except (KeyError, ValueError):
            raise exceptions.RefreshError("Unable to parse subject_token from JSON file '{}' using key '{}'".format(filename, subject_token_field_name))
    if not token:
        raise exceptions.RefreshError('Missing subject_token in the credential_source file')
    return token