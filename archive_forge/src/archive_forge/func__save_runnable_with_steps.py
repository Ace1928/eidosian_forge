import os
from pathlib import Path
from typing import Union
import cloudpickle
import yaml
from mlflow.exceptions import MlflowException
from mlflow.langchain.utils import (
def _save_runnable_with_steps(model, file_path: Union[Path, str], loader_fn=None, persist_dir=None):
    """Save the model with steps. Currently it supports saving RunnableSequence and
    RunnableParallel.

    If saving a RunnableSequence, steps is a list of Runnable objects. We save each step to the
    subfolder named by the step index.
    e.g.  - model
            - steps
              - 0
                - model.yaml
              - 1
                - model.pkl
            - steps.yaml
    If saving a RunnableParallel, steps is a dictionary of key-Runnable pairs. We save each step to
    the subfolder named by the key.
    e.g.  - model
            - steps
              - context
                - model.yaml
              - question
                - model.pkl
            - steps.yaml

    We save steps.yaml file to the model folder. It contains each step's model's configuration.

    Args:
        model: Runnable to be saved.
        file_path: Path to file to save the model to.
    """
    save_path = Path(file_path)
    save_path.mkdir(parents=True, exist_ok=True)
    steps_path = save_path / _STEPS_FOLDER_NAME
    steps_path.mkdir()
    steps = model.steps
    if isinstance(steps, list):
        generator = enumerate(steps)
    elif isinstance(steps, dict):
        generator = steps.items()
    else:
        raise MlflowException(f'Runnable {model} steps attribute must be either a list or a dictionary. Got {type(steps).__name__}.')
    unsaved_runnables = {}
    steps_conf = {}
    for key, runnable in generator:
        step = str(key)
        save_runnable_path = steps_path / step
        save_runnable_path.mkdir()
        try:
            steps_conf[step] = _save_internal_runnables(runnable, save_runnable_path, loader_fn, persist_dir)
        except Exception as e:
            unsaved_runnables[step] = f'{runnable} -- {e}'
    if unsaved_runnables:
        raise MlflowException(f'Failed to save runnable sequence: {unsaved_runnables}.')
    with save_path.joinpath(_RUNNABLE_STEPS_FILE_NAME).open('w') as f:
        yaml.dump(steps_conf, f, default_flow_style=False)