import os
from typing import Any, Dict, Optional
import wandb
from triad import assert_or_throw
from tune import parse_logger
from tune.concepts.logger import MetricLogger
from tune.exceptions import TuneRuntimeError
from wandb.env import get_project
from wandb.sdk.lib.apikey import api_key
from wandb.wandb_run import Run
class WandbProjectLogger(WandbLoggerBase):

    def __init__(self, name: str):
        super().__init__(None)
        self._name = name or get_project()

    @property
    def project_name(self) -> str:
        return self._name

    def create_child(self, name: str=None, description: Optional[str]=None, is_step: bool=False) -> MetricLogger:
        assert_or_throw(not is_step, ValueError("can't create step logger"))
        return WandbGroupLogger(self.project_name, name)