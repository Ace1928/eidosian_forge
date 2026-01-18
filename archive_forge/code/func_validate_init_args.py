import logging
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import requests
from langchain_core.documents import Document
from tenacity import (
from langchain_community.document_loaders.base import BaseLoader
@staticmethod
def validate_init_args(url: Optional[str]=None, api_key: Optional[str]=None, username: Optional[str]=None, session: Optional[requests.Session]=None, oauth2: Optional[dict]=None, token: Optional[str]=None) -> Union[List, None]:
    """Validates proper combinations of init arguments"""
    errors = []
    if url is None:
        errors.append('Must provide `base_url`')
    if api_key and (not username) or (username and (not api_key)):
        errors.append('If one of `api_key` or `username` is provided, the other must be as well.')
    non_null_creds = list((x is not None for x in (api_key or username, session, oauth2, token)))
    if sum(non_null_creds) > 1:
        all_names = ('(api_key, username)', 'session', 'oath2', 'token')
        provided = tuple((n for x, n in zip(non_null_creds, all_names) if x))
        errors.append(f'Cannot provide a value for more than one of: {all_names}. Received values for: {provided}')
    if oauth2 and set(oauth2.keys()) != {'access_token', 'access_token_secret', 'consumer_key', 'key_cert'}:
        errors.append("You have either omitted require keys or added extra keys to the oauth2 dictionary. key values should be `['access_token', 'access_token_secret', 'consumer_key', 'key_cert']`")
    return errors or None