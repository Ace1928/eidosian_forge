import json
import os
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
import yaml
from .logging import get_logger
from .state import PartialState
from .utils import (
class ClearMLTracker(GeneralTracker):
    """
    A `Tracker` class that supports `clearml`. Should be initialized at the start of your script.

    Args:
        run_name (`str`, *optional*):
            Name of the experiment. Environment variables `CLEARML_PROJECT` and `CLEARML_TASK` have priority over this
            argument.
        kwargs:
            Kwargs passed along to the `Task.__init__` method.
    """
    name = 'clearml'
    requires_logging_directory = False

    @on_main_process
    def __init__(self, run_name: str=None, **kwargs):
        from clearml import Task
        current_task = Task.current_task()
        self._initialized_externally = False
        if current_task:
            self._initialized_externally = True
            self.task = current_task
            return
        kwargs.setdefault('project_name', os.environ.get('CLEARML_PROJECT', run_name))
        kwargs.setdefault('task_name', os.environ.get('CLEARML_TASK', run_name))
        self.task = Task.init(**kwargs)

    @property
    def tracker(self):
        return self.task

    @on_main_process
    def store_init_configuration(self, values: dict):
        """
        Connect configuration dictionary to the Task object. Should be run at the beginning of your experiment.

        Args:
            values (`dict`):
                Values to be stored as initial hyperparameters as key-value pairs.
        """
        return self.task.connect_configuration(values)

    @on_main_process
    def log(self, values: Dict[str, Union[int, float]], step: Optional[int]=None, **kwargs):
        """
        Logs `values` dictionary to the current run. The dictionary keys must be strings. The dictionary values must be
        ints or floats

        Args:
            values (`Dict[str, Union[int, float]]`):
                Values to be logged as key-value pairs. If the key starts with 'eval_'/'test_'/'train_', the value will
                be reported under the 'eval'/'test'/'train' series and the respective prefix will be removed.
                Otherwise, the value will be reported under the 'train' series, and no prefix will be removed.
            step (`int`, *optional*):
                If specified, the values will be reported as scalars, with the iteration number equal to `step`.
                Otherwise they will be reported as single values.
            kwargs:
                Additional key word arguments passed along to the `clearml.Logger.report_single_value` or
                `clearml.Logger.report_scalar` methods.
        """
        clearml_logger = self.task.get_logger()
        for k, v in values.items():
            if not isinstance(v, (int, float)):
                logger.warning_once(f'''Accelerator is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a scalar. This invocation of ClearML logger's  report_scalar() is incorrect so we dropped this attribute.''')
                continue
            if step is None:
                clearml_logger.report_single_value(name=k, value=v, **kwargs)
                continue
            title, series = ClearMLTracker._get_title_series(k)
            clearml_logger.report_scalar(title=title, series=series, value=v, iteration=step, **kwargs)

    @on_main_process
    def log_images(self, values: dict, step: Optional[int]=None, **kwargs):
        """
        Logs `images` to the current run.

        Args:
            values (`Dict[str, List[Union[np.ndarray, PIL.Image]]`):
                Values to be logged as key-value pairs. The values need to have type `List` of `np.ndarray` or
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `clearml.Logger.report_image` method.
        """
        clearml_logger = self.task.get_logger()
        for k, v in values.items():
            title, series = ClearMLTracker._get_title_series(k)
            clearml_logger.report_image(title=title, series=series, iteration=step, image=v, **kwargs)

    @on_main_process
    def log_table(self, table_name: str, columns: List[str]=None, data: List[List[Any]]=None, dataframe: Any=None, step: Optional[int]=None, **kwargs):
        """
        Log a Table to the task. Can be defined eitherwith `columns` and `data` or with `dataframe`.

        Args:
            table_name (`str`):
                The name of the table
            columns (list of `str`, *optional*):
                The name of the columns on the table
            data (List of List of Any data type, *optional*):
                The data to be logged in the table. If `columns` is not specified, then the first entry in data will be
                the name of the columns of the table
            dataframe (Any data type, *optional*):
                The data to be logged in the table
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `clearml.Logger.report_table` method.
        """
        to_report = dataframe
        if dataframe is None:
            if data is None:
                raise ValueError('`ClearMLTracker.log_table` requires that `data` to be supplied if `dataframe` is `None`')
            to_report = [columns] + data if columns else data
        title, series = ClearMLTracker._get_title_series(table_name)
        self.task.get_logger().report_table(title=title, series=series, table_plot=to_report, iteration=step, **kwargs)

    @on_main_process
    def finish(self):
        """
        Close the ClearML task. If the task was initialized externally (e.g. by manually calling `Task.init`), this
        function is a noop
        """
        if self.task and (not self._initialized_externally):
            self.task.close()

    @staticmethod
    def _get_title_series(name):
        for prefix in ['eval', 'test', 'train']:
            if name.startswith(prefix + '_'):
                return (name[len(prefix) + 1:], prefix)
        return (name, 'train')