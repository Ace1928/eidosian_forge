from .. import utils
from .._lazyload import requests
import os
import tempfile
import urllib.request
import zipfile
@utils._with_pkg(pkg='requests')
def _GET_google_drive(id):
    """Post a GET request to Google Drive."""
    global _GOOGLE_DRIVE_URL
    with requests.Session() as session:
        response = session.get(_GOOGLE_DRIVE_URL, params={'id': id}, stream=True)
        token = _google_drive_confirm_token(response)
        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(_GOOGLE_DRIVE_URL, params=params, stream=True)
    return response