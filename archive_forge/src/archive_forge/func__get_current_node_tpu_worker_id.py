import os
import re
import glob
import requests
import logging
from typing import Dict, Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
@staticmethod
def _get_current_node_tpu_worker_id() -> Optional[int]:
    """Return the worker index of the TPU pod."""
    try:
        worker_id = os.getenv(GKE_TPU_WORKER_ID_ENV_VAR, None)
        if not worker_id:
            worker_id = _get_tpu_metadata(key=GCE_TPU_WORKER_ID_KEY)
        if worker_id:
            return int(worker_id)
        else:
            return None
    except ValueError as e:
        logging.debug('Could not get TPU worker id: %s', e)
        return None