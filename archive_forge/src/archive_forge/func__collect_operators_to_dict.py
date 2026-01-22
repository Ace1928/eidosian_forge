import json
import re
import threading
from typing import Dict
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators.map_operator import AbstractUDFMap
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.logical.operators.write_operator import Write
def _collect_operators_to_dict(op: LogicalOperator, ops_dict: Dict[str, int]):
    """Collect the logical operator name and count into `ops_dict`."""
    for child in op.input_dependencies:
        _collect_operators_to_dict(child, ops_dict)
    op_name = op.name
    if isinstance(op, Read):
        op_name = f'Read{op._datasource.get_name()}'
        if op_name not in _op_name_white_list:
            op_name = 'ReadCustom'
    elif isinstance(op, Write):
        op_name = f'Write{op._datasink_or_legacy_datasource.get_name()}'
        if op_name not in _op_name_white_list:
            op_name = 'WriteCustom'
    elif isinstance(op, AbstractUDFMap):
        op_name = re.sub('\\(.*\\)$', '', op_name)
    if op_name not in _op_name_white_list:
        op_name = 'Unknown'
    ops_dict.setdefault(op_name, 0)
    ops_dict[op_name] += 1