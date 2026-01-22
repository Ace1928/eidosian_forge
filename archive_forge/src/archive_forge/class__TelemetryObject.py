import re
import sys
from types import TracebackType
from typing import TYPE_CHECKING, ContextManager, Dict, List, Optional, Set, Type
import wandb
from wandb.proto.wandb_telemetry_pb2 import Imports as TelemetryImports
from wandb.proto.wandb_telemetry_pb2 import TelemetryRecord
class _TelemetryObject:
    _run: Optional['wandb_run.Run']
    _obj: TelemetryRecord

    def __init__(self, run: Optional['wandb_run.Run']=None, obj: Optional[TelemetryRecord]=None) -> None:
        self._run = run or wandb.run
        self._obj = obj or TelemetryRecord()

    def __enter__(self) -> TelemetryRecord:
        return self._obj

    def __exit__(self, exctype: Optional[Type[BaseException]], excinst: Optional[BaseException], exctb: Optional[TracebackType]) -> None:
        if not self._run:
            return
        self._run._telemetry_callback(self._obj)