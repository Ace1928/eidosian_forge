import atexit
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
import psutil
import wandb
from wandb import env, trigger
from wandb.errors import Error
from wandb.sdk.lib.exit_hooks import ExitHooks
from wandb.sdk.lib.import_hooks import unregister_all_post_import_hooks
def _service_connect(self) -> None:
    port = self._token.port
    svc_iface = self._get_service_interface()
    try:
        svc_iface._svc_connect(port=port)
    except ConnectionRefusedError as e:
        if not psutil.pid_exists(self._token.pid):
            message = 'Connection to wandb service failed since the process is not available. '
        else:
            message = f'Connection to wandb service failed: {e}. '
        raise ManagerConnectionRefusedError(message)
    except Exception as e:
        raise ManagerConnectionError(f'Connection to wandb service failed: {e}')