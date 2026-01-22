import logging
import os
from argparse import Namespace
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import Module
from typing_extensions import override
from lightning_fabric.loggers.logger import Logger, rank_zero_experiment
from lightning_fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning_fabric.utilities.logger import _add_prefix, _convert_params, _flatten_dict
from lightning_fabric.utilities.logger import _sanitize_params as _utils_sanitize_params
from lightning_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning_fabric.utilities.types import _PATH
from lightning_fabric.wrappers import _unwrap_objects
def _get_next_version(self) -> int:
    save_dir = os.path.join(self.root_dir, self.name)
    try:
        listdir_info = self._fs.listdir(save_dir)
    except OSError:
        log.warning('Missing logger folder: %s', save_dir)
        return 0
    existing_versions = []
    for listing in listdir_info:
        d = listing['name']
        bn = os.path.basename(d)
        if _is_dir(self._fs, d) and bn.startswith('version_'):
            dir_ver = bn.split('_')[1].replace('/', '')
            if dir_ver.isdigit():
                existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0
    return max(existing_versions) + 1