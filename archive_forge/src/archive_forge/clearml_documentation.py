from typing import Dict, Any, Tuple, Callable, List, Optional, IO
from types import ModuleType
import os
import sys
from spacy import Language
from spacy.util import SimpleFrozenList
from .util import dict_to_dot, dot_to_dict, matcher_for_regex_patterns
from .util import setup_default_console_logger, LoggerT
Creates a logger that interoperates with the ClearML framework.

    Args:
        project_name (str):
            The name of the project in the ClearML interface. The project will be created automatically if it doesn't exist yet.
        task_name (str):
            The name of the ClearML task. A task is an experiment that lives inside a project. Can be non-unique.
        remove_config_values (List[str]):
            A list of values to exclude from the config before it is uploaded to ClearML. Defaults to [].
        model_log_interval (Optional[int]):
            Steps to wait between logging model checkpoints to the ClearML dasboard (default: `None`). Will have no effect without also setting `log_best_dir` or `log_latest_dir`. Defaults to None.
        log_dataset_dir (Optional[str]):
            Directory containing the dataset to be logged and versioned as a ClearML Dataset. Defaults to None.
        log_best_dir (Optional[str]):
            Directory containing the best trained model as saved by spaCy, to be logged and versioned as a ClearML artifact. Defaults to None.
        log_latest_dir (Optional[str]):
            Directory containing the latest trained model as saved by spaCy, to be logged and versioned as a ClearML artifact. Defaults to None.

    Returns:
        LoggerT: Logger instance.
    