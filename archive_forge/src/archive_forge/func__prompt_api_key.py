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
def _prompt_api_key(self) -> Tuple[Optional[str], ApiKeyStatus]:
    api = Api(self._settings)
    while True:
        try:
            key = apikey.prompt_api_key(self._settings, api=api, no_offline=self._settings.force if self._settings else None, no_create=self._settings.force if self._settings else None)
        except ValueError as e:
            wandb.termerror(e.args[0])
            continue
        except TimeoutError:
            wandb.termlog('W&B disabled due to login timeout.')
            return (None, ApiKeyStatus.DISABLED)
        if key is False:
            return (None, ApiKeyStatus.NOTTY)
        if not key:
            return (None, ApiKeyStatus.OFFLINE)
        return (key, ApiKeyStatus.VALID)