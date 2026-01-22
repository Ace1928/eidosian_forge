import io
import json
import os
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
def _get_token_data(self, request):
    if self._credential_source_file:
        return self._get_file_data(self._credential_source_file)
    else:
        return self._get_url_data(request, self._credential_source_url, self._credential_source_headers)