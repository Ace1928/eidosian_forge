import json
from typing import Any, Dict, NewType, Optional, Sequence
from wandb.proto import wandb_internal_pb2
from wandb.sdk.lib import proto_util, telemetry
def _add_unset_keys_from_subtree(self, old_config_tree: Dict[str, Any], path: Sequence[str]) -> None:
    """Uses the given subtree for keys that aren't already set."""
    old_subtree = _subtree(old_config_tree, path, create=False)
    if not old_subtree:
        return
    new_subtree = _subtree(self._tree, path, create=True)
    assert new_subtree is not None
    for key, value in old_subtree.items():
        if key not in new_subtree:
            new_subtree[key] = value