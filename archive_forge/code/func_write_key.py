import os
import stat
import sys
import textwrap
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union
from urllib.parse import urlparse
import click
import requests.utils
import wandb
from wandb.apis import InternalApi
from wandb.errors import term
from wandb.util import _is_databricks, isatty, prompt_choices
from .wburls import wburls
def write_key(settings: 'Settings', key: Optional[str], api: Optional['InternalApi']=None, anonymous: bool=False) -> None:
    if not key:
        raise ValueError('No API key specified.')
    api = api or InternalApi()
    _, suffix = key.split('-', 1) if '-' in key else ('', key)
    if len(suffix) != 40:
        raise ValueError('API key must be 40 characters long, yours was %s' % len(key))
    if anonymous:
        api.set_setting('anonymous', 'true', globally=True, persist=True)
    else:
        api.clear_setting('anonymous', globally=True, persist=True)
    write_netrc(settings.base_url, 'user', key)