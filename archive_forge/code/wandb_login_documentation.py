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
Set up W&B login credentials.

    By default, this will only store the credentials locally without
    verifying them with the W&B server. To verify credentials, pass
    verify=True.

    Arguments:
        anonymous: (string, optional) Can be "must", "allow", or "never".
            If set to "must" we'll always log in anonymously, if set to
            "allow" we'll only create an anonymous user if the user
            isn't already logged in.
        key: (string, optional) authentication key.
        relogin: (bool, optional) If true, will re-prompt for API key.
        host: (string, optional) The host to connect to.
        force: (bool, optional) If true, will force a relogin.
        timeout: (int, optional) Number of seconds to wait for user input.
        verify: (bool) Verify the credentials with the W&B server.

    Returns:
        bool: if key is configured

    Raises:
        AuthenticationError - if api_key fails verification with the server
        UsageError - if api_key cannot be configured and no tty
    