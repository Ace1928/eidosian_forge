import contextlib
import platform
import uuid
import warnings
import weakref
from collections import defaultdict
from itertools import count
from typing import (
from weakref import ReferenceType
import torch
import torch.fx.traceback as fx_traceback
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
def unpack_error_cb(e: CheckpointError):

    def get_str_tb(label, capture_logs):
        out = ''
        total_len = len(capture_logs.logs)
        for i, (log, tb) in enumerate(zip(capture_logs.logs, capture_logs.tbs)):
            out += f'{log}   ({i + 1} of {total_len} in {label})\n\n'
            found_torch_dispatch = False
            for line in tb:
                is_torch_dispatch = line['name'] == '__torch_dispatch__'
                if not found_torch_dispatch and (not is_torch_dispatch):
                    continue
                elif is_torch_dispatch:
                    found_torch_dispatch = True
                    continue
                out += f'{line['filename']}:{line['line']}:{line['name']}\n'
            out += '\n\n'
        return out
    assert capture_logs_fwd.logs is not None
    assert capture_logs_recompute.logs is not None
    raise CheckpointError(_checkpoint_error_template.format(forward_traces=get_str_tb('original', capture_logs_fwd), recompute_traces=get_str_tb('recompute', capture_logs_recompute), forward_ops='\n'.join(capture_logs_fwd.logs), recompute_ops='\n'.join(capture_logs_recompute.logs))) from e