import atexit
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
import psutil
import wandb
from wandb import env, trigger
from wandb.errors import Error
from wandb.sdk.lib.exit_hooks import ExitHooks
from wandb.sdk.lib.import_hooks import unregister_all_post_import_hooks
def _inform_attach(self, attach_id: str) -> Optional[Dict[str, Any]]:
    svc_iface = self._get_service_interface()
    try:
        response = svc_iface._svc_inform_attach(attach_id=attach_id)
    except Exception:
        return None
    return response.settings