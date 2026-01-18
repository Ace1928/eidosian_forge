import atexit
import functools
import os
import pathlib
import sys
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type, Union
from urllib.parse import quote
import sentry_sdk  # type: ignore
import sentry_sdk.utils  # type: ignore
import wandb
import wandb.env
import wandb.util
Configure the Sentry scope for the current thread.

        This function should be called at the beginning of every thread that
        will send events to Sentry. It sets the tags that will be applied to
        all events sent from this thread. It also tries to start a session
        if one doesn't already exist for this thread.
        