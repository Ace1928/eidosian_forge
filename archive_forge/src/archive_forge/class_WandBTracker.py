import json
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
import yaml
from .logging import get_logger
from .state import PartialState
from .utils import (
class WandBTracker(GeneralTracker):
    """
    A `Tracker` class that supports `wandb`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run.
        kwargs:
            Additional key word arguments passed along to the `wandb.init` method.
    """
    name = 'wandb'
    requires_logging_directory = False
    main_process_only = False

    @on_main_process
    def __init__(self, run_name: str, **kwargs):
        super().__init__()
        self.run_name = run_name
        import wandb
        self.run = wandb.init(project=self.run_name, **kwargs)
        logger.debug(f'Initialized WandB project {self.run_name}')
        logger.debug('Make sure to log any initial configurations with `self.store_init_configuration` before training!')

    @property
    def tracker(self):
        return self.run

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        import wandb
        wandb.config.update(values, allow_val_change=True)
        logger.debug('Stored initial configuration hyperparameters to WandB')

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
                Additional key word arguments passed along to the `wandb.log` method.
        """
        self.run.log(values, step=step, **kwargs)
        logger.debug('Successfully logged to WandB')

    @on_main_process
    def log_images(self, values: dict, step: Optional[int]=None, **kwargs):
        """
        Logs `images` to the current run.

        Args:
            values (Dictionary `str` to `List` of `np.ndarray` or `PIL.Image`):
                Values to be logged as key-value pairs. The values need to have type `List` of `np.ndarray` or
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `wandb.log` method.
        """
        import wandb
        for k, v in values.items():
            self.log({k: [wandb.Image(image) for image in v]}, step=step, **kwargs)
        logger.debug('Successfully logged images to WandB')

    @on_main_process
    def log_table(self, table_name: str, columns: List[str]=None, data: List[List[Any]]=None, dataframe: Any=None, step: Optional[int]=None, **kwargs):
        """
        Log a Table containing any object type (text, image, audio, video, molecule, html, etc). Can be defined either
        with `columns` and `data` or with `dataframe`.

        Args:
            table_name (`str`):
                The name to give to the logged table on the wandb workspace
            columns (list of `str`, *optional*):
                The name of the columns on the table
            data (List of List of Any data type, *optional*):
                The data to be logged in the table
            dataframe (Any data type, *optional*):
                The data to be logged in the table
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
        """
        import wandb
        values = {table_name: wandb.Table(columns=columns, data=data, dataframe=dataframe)}
        self.log(values, step=step, **kwargs)

    @on_main_process
    def finish(self):
        """
        Closes `wandb` writer
        """
        self.run.finish()
        logger.debug('WandB run closed')