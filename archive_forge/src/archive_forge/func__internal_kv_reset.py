from typing import List, Optional, Union
from ray._private.client_mode_hook import client_mode_hook
from ray._raylet import GcsClient
def _internal_kv_reset():
    global global_gcs_client, _initialized
    global_gcs_client = None
    _initialized = False