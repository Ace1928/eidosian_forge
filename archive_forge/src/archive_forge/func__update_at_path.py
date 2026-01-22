import json
from typing import Any, Dict, NewType, Optional, Sequence
from wandb.proto import wandb_internal_pb2
from wandb.sdk.lib import proto_util, telemetry
def _update_at_path(self, key_path: Sequence[str], value: Any) -> None:
    """Sets the value at the path in the config tree."""
    subtree = _subtree(self._tree, key_path[:-1], create=True)
    assert subtree is not None
    subtree[key_path[-1]] = value