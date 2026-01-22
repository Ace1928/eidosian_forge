import atexit
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
import psutil
import wandb
from wandb import env, trigger
from wandb.errors import Error
from wandb.sdk.lib.exit_hooks import ExitHooks
from wandb.sdk.lib.import_hooks import unregister_all_post_import_hooks
def _inform_start(self, settings: 'wandb_settings_pb2.Settings', run_id: str) -> None:
    svc_iface = self._get_service_interface()
    svc_iface._svc_inform_start(settings=settings, run_id=run_id)