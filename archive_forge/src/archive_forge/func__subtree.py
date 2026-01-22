import json
from typing import Any, Dict, NewType, Optional, Sequence
from wandb.proto import wandb_internal_pb2
from wandb.sdk.lib import proto_util, telemetry
def _subtree(tree: Dict[str, Any], key_path: Sequence[str], *, create: bool=False) -> Optional[Dict[str, Any]]:
    """Returns a subtree at the given path."""
    for key in key_path:
        subtree = tree.get(key)
        if not subtree:
            if create:
                subtree = {}
                tree[key] = subtree
            else:
                return None
        tree = subtree
    return tree