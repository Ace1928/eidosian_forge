import os
import re
import glob
import requests
import logging
from typing import Dict, Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
@staticmethod
def _get_current_node_tpu_pod_type() -> Optional[str]:
    """Get the TPU pod type of the current node if applicable.

        Individual TPU VMs within a TPU pod must know what type
        of pod it is a part of. This is necessary for the
        ML framework to work properly.

        The logic is different if the TPU was provisioned via:
        ```
        gcloud tpus tpu-vm create ...
        ```
        (i.e. a GCE VM), vs through GKE:
        - GCE VMs will always have a metadata server to poll this info
        - GKE VMS will have environment variables preset.

        Returns:
            A string representing the current TPU pod type, e.g.
            v4-16.

        """
    accelerator_type = os.getenv(GKE_TPU_ACCELERATOR_TYPE_ENV_VAR, '')
    if not accelerator_type:
        accelerator_type = _get_tpu_metadata(key=GCE_TPU_ACCELERATOR_KEY)
    if accelerator_type and TPUAcceleratorManager.is_valid_tpu_accelerator_type(tpu_accelerator_type=accelerator_type):
        return accelerator_type
    logging.debug('Failed to get a valid accelerator type.')
    return None