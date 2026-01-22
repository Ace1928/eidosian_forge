import io
import json
import os
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
def _get_file_data(self, filename):
    if not os.path.exists(filename):
        raise exceptions.RefreshError("File '{}' was not found.".format(filename))
    with io.open(filename, 'r', encoding='utf-8') as file_obj:
        return (file_obj.read(), filename)