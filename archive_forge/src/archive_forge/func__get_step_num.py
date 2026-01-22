import logging
from typing import Dict
from typing_extensions import override
from lightning_fabric.accelerators.xla import _XLA_AVAILABLE
from pytorch_lightning.profilers.profiler import Profiler
def _get_step_num(self, action_name: str) -> int:
    if action_name not in self._step_recoding_map:
        self._step_recoding_map[action_name] = 1
    else:
        self._step_recoding_map[action_name] += 1
    return self._step_recoding_map[action_name]