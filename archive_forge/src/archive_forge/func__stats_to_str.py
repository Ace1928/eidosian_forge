import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, TextIO, Union
from lightning_fabric.utilities.cloud_io import get_filesystem
def _stats_to_str(self, stats: Dict[str, str]) -> str:
    stage = f'{self._stage.upper()} ' if self._stage is not None else ''
    output = [stage + 'Profiler Report']
    for action, value in stats.items():
        header = f'Profile stats for: {action}'
        if self._local_rank is not None:
            header += f' rank: {self._local_rank}'
        output.append(header)
        output.append(value)
    return os.linesep.join(output)