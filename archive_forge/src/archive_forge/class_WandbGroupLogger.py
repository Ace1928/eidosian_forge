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
class WandbGroupLogger(WandbLoggerBase):

    def __init__(self, project_name: str, group: Optional[str]=None):
        super().__init__(None)
        self._project_name = project_name
        self._group = group or self.unique_id

    def __getstate__(self) -> Dict[str, Any]:
        return dict(project_name=self.project_name, group=self.group, api_key=self.api_key, unique_id=self.unique_id)

    def __setstate__(self, data: Dict[str, Any]) -> None:
        self._unique_id = data['unique_id']
        os.environ['WANDB_SILENT'] = 'true'
        wandb.login(key=data['api_key'], relogin=True)
        self._project_name = data['project_name']
        self._group = data['group']
        self._run = None

    @property
    def project_name(self) -> str:
        return self._project_name

    @property
    def group(self) -> str:
        return self._group

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        pass

    def create_child(self, name: str=None, description: Optional[str]=None, is_step: bool=False) -> MetricLogger:
        if is_step:
            return MetricLogger()
        else:
            os.environ['WANDB_SILENT'] = 'true'
            os.environ['WANDB_START_METHOD'] = 'thread'
            run = wandb.init(project=self.project_name, settings={'silent': True}, group=self.group, name=name, notes=description, reinit=True, allow_val_change=True)
            return WandbLogger(run)