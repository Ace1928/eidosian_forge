import functools
import itertools
import logging
import os
import re
from dataclasses import dataclass, field
from importlib import __import__
from typing import Dict, List, Optional, Set, Union
from weakref import WeakSet
import torch._guards
import torch.distributed as dist
class TorchLogsFormatter(logging.Formatter):

    def format(self, record):
        artifact_name = getattr(logging.getLogger(record.name), 'artifact_name', None)
        if artifact_name is not None:
            artifact_formatter = log_registry.artifact_log_formatters.get(artifact_name, None)
            if artifact_formatter is not None:
                return artifact_formatter.format(record)
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)
        s = record.message
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != '\n':
                s = s + '\n'
            s = s + record.exc_text
        if record.stack_info:
            if s[-1:] != '\n':
                s = s + '\n'
            s = s + self.formatStack(record.stack_info)
        lines = s.split('\n')
        record.rankprefix = ''
        if dist.is_available() and dist.is_initialized():
            record.rankprefix = f'[rank{dist.get_rank()}]:'
        record.traceid = ''
        if (trace_id := torch._guards.CompileContext.current_trace_id()) is not None:
            record.traceid = f' [{trace_id}]'
        prefix = f'{record.rankprefix}[{record.asctime}]{record.traceid} {record.name}: [{record.levelname}]'
        return '\n'.join((f'{prefix} {l}' for l in lines))