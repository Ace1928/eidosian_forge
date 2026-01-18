import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
def update_configs(self, node_configs: Dict[str, Any], skip_content_hash: bool) -> None:
    self._node_configs = prepare_config(node_configs)
    if skip_content_hash:
        return
    self._calculate_hashes()
    self._sync_continuously = self._node_configs.get('generate_file_mounts_contents_hash', True)