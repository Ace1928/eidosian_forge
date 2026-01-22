from __future__ import annotations
import hashlib
import os
import typing
import urllib.parse
import warnings
from dataclasses import dataclass, field
import fastapi
from fastapi.responses import RedirectResponse
from huggingface_hub import HfFolder, whoami
from .utils import get_space
def _generate_redirect_uri(request: fastapi.Request) -> str:
    if '_target_url' in request.query_params:
        target = request.query_params['_target_url']
    else:
        target = '/?' + urllib.parse.urlencode(request.query_params)
    redirect_uri = request.url_for('oauth_redirect_callback').include_query_params(_target_url=target)
    redirect_uri_as_str = str(redirect_uri)
    if redirect_uri.netloc.endswith('.hf.space'):
        redirect_uri_as_str = redirect_uri_as_str.replace('http://', 'https://')
    return redirect_uri_as_str