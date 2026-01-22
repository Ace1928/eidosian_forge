import copy
from datetime import datetime
from typing import Callable, Dict, Optional, Union
from packaging import version
import wandb
from wandb.sdk.lib import telemetry
def _make_predictor(self, model: YOLO):
    overrides = copy.deepcopy(model.overrides)
    overrides['conf'] = 0.1
    self.predictor = self.task_map[self.task]['predictor'](overrides=overrides)
    self.predictor.callbacks = {}
    self.predictor.args.save = False
    self.predictor.args.save_txt = False
    self.predictor.args.save_crop = False
    self.predictor.args.verbose = None