import json
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
import yaml
from .logging import get_logger
from .state import PartialState
from .utils import (
class CometMLTracker(GeneralTracker):
    """
    A `Tracker` class that supports `comet_ml`. Should be initialized at the start of your script.

    API keys must be stored in a Comet config file.

    Args:
        run_name (`str`):
            The name of the experiment run.
        kwargs:
            Additional key word arguments passed along to the `Experiment.__init__` method.
    """
    name = 'comet_ml'
    requires_logging_directory = False

    @on_main_process
    def __init__(self, run_name: str, **kwargs):
        super().__init__()
        self.run_name = run_name
        from comet_ml import Experiment
        self.writer = Experiment(project_name=run_name, **kwargs)
        logger.debug(f'Initialized CometML project {self.run_name}')
        logger.debug('Make sure to log any initial configurations with `self.store_init_configuration` before training!')

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        self.writer.log_parameters(values)
        logger.debug('Stored initial configuration hyperparameters to CometML')

    @on_main_process
    def log(self, values: dict, step: Optional[int]=None, **kwargs):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to either `Experiment.log_metric`, `Experiment.log_other`,
                or `Experiment.log_metrics` method based on the contents of `values`.
        """
        if step is not None:
            self.writer.set_step(step)
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.log_metric(k, v, step=step, **kwargs)
            elif isinstance(v, str):
                self.writer.log_other(k, v, **kwargs)
            elif isinstance(v, dict):
                self.writer.log_metrics(v, step=step, **kwargs)
        logger.debug('Successfully logged to CometML')

    @on_main_process
    def finish(self):
        """
        Closes `comet-ml` writer
        """
        self.writer.end()
        logger.debug('CometML run closed')