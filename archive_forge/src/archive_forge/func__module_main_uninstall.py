import importlib.machinery
import logging
import multiprocessing
import os
import queue
import sys
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union, cast
import wandb
from wandb.sdk.interface.interface import InterfaceBase
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal.internal import wandb_internal
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib.mailbox import Mailbox
from wandb.sdk.wandb_manager import _Manager
from wandb.sdk.wandb_settings import Settings
def _module_main_uninstall(self) -> None:
    main_module = sys.modules['__main__']
    main_module.__spec__ = self._save_mod_spec
    if self._save_mod_path:
        main_module.__file__ = self._save_mod_path