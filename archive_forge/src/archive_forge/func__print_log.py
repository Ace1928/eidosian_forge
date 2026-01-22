import json
import logging
from datetime import datetime
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple
import click
import yaml
import ray._private.services as services
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.state import (
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException
from ray.util.annotations import PublicAPI
def _print_log(address: Optional[str]=None, node_id: Optional[str]=None, node_ip: Optional[str]=None, filename: Optional[str]=None, actor_id: Optional[str]=None, pid: Optional[int]=None, follow: bool=False, tail: int=DEFAULT_LOG_LIMIT, timeout: int=DEFAULT_RPC_TIMEOUT, interval: Optional[float]=None, suffix: str='out', encoding: str='utf-8', encoding_errors: str='strict', task_id: Optional[str]=None, attempt_number: int=0, submission_id: Optional[str]=None):
    """Wrapper around `get_log()` that prints the preamble and the log lines"""
    if tail > 0:
        print(f'--- Log has been truncated to last {tail} lines. Use `--tail` flag to toggle. Set to -1 for getting the entire file. ---\n')
    if node_id is None and node_ip is None:
        node_ip = _get_head_node_ip(address)
    for chunk in get_log(address=address, node_id=node_id, node_ip=node_ip, filename=filename, actor_id=actor_id, tail=tail, pid=pid, follow=follow, _interval=interval, timeout=timeout, suffix=suffix, encoding=encoding, errors=encoding_errors, task_id=task_id, attempt_number=attempt_number, submission_id=submission_id):
        print(chunk, end='', flush=True)