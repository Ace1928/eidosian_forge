import json
from typing import Any, Dict, NewType, Optional, Sequence
from wandb.proto import wandb_internal_pb2
from wandb.sdk.lib import proto_util, telemetry
def _delete_at_path(self, key_path: Sequence[str]) -> None:
    """Removes the subtree at the path in the config tree."""
    subtree = _subtree(self._tree, key_path[:-1], create=False)
    if subtree:
        del subtree[key_path[-1]]