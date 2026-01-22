import enum
import os
import sys
from typing import Dict, Optional, Tuple
import click
import wandb
from wandb.errors import AuthenticationError, UsageError
from wandb.old.settings import Settings as OldSettings
from ..apis import InternalApi
from .internal.internal_api import Api
from .lib import apikey
from .wandb_settings import Settings, Source
def _login(anonymous: Optional[Literal['must', 'allow', 'never']]=None, key: Optional[str]=None, relogin: Optional[bool]=None, host: Optional[str]=None, force: Optional[bool]=None, timeout: Optional[int]=None, _backend=None, _silent: Optional[bool]=None, _disable_warning: Optional[bool]=None, _entity: Optional[str]=None):
    kwargs = dict(locals())
    _disable_warning = kwargs.pop('_disable_warning', None)
    if wandb.run is not None:
        if not _disable_warning:
            wandb.termwarn('Calling wandb.login() after wandb.init() has no effect.')
        return True
    wlogin = _WandbLogin()
    _backend = kwargs.pop('_backend', None)
    if _backend:
        wlogin.set_backend(_backend)
    _silent = kwargs.pop('_silent', None)
    if _silent:
        wlogin.set_silent(_silent)
    _entity = kwargs.pop('_entity', None)
    if _entity:
        wlogin.set_entity(_entity)
    wlogin.setup(kwargs)
    if wlogin._settings._offline:
        return False
    elif wandb.util._is_kaggle() and (not wandb.util._has_internet()):
        wandb.termerror('To use W&B in kaggle you must enable internet in the settings panel on the right.')
        return False
    logged_in = wlogin.login()
    key = kwargs.get('key')
    if key:
        wlogin.configure_api_key(key)
    if logged_in:
        return logged_in
    if not key:
        wlogin.prompt_api_key()
    wlogin.propogate_login()
    return wlogin._key or False