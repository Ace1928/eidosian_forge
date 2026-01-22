import abc
import dataclasses
import os
from typing import Any, Dict, List, TYPE_CHECKING
import cirq
from cirq import _compat
from cirq.protocols import dataclass_json_dict
def _update_updatable_files(egr_record: ExecutableGroupResultFilesystemRecord, shared_rt_info: 'cg.SharedRuntimeInfo', data_dir: str):
    """Safely update ExecutableGroupResultFilesystemRecord.json.gz and SharedRuntimeInfo.json.gz
    during an execution run.
    """
    _safe_to_json(shared_rt_info, part_path=f'{data_dir}/SharedRuntimeInfo.json.gz.part', nominal_path=f'{data_dir}/SharedRuntimeInfo.json.gz', bak_path=f'{data_dir}/SharedRuntimeInfo.json.gz.bak')
    _safe_to_json(egr_record, part_path=f'{data_dir}/ExecutableGroupResultFilesystemRecord.json.gz.part', nominal_path=f'{data_dir}/ExecutableGroupResultFilesystemRecord.json.gz', bak_path=f'{data_dir}/ExecutableGroupResultFilesystemRecord.json.gz.bak')