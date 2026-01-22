from typing import Dict, Any, Tuple, Callable, List, Optional, IO
from types import ModuleType
import os
import sys
from spacy import Language, load
from spacy.util import SimpleFrozenList
from .util import dict_to_dot, dot_to_dict, matcher_for_regex_patterns
from .util import setup_default_console_logger, LoggerT
def _setup_mlflow(mlflow: ModuleType, nlp: Language, run_id: Optional[str]=None, experiment_id: Optional[str]=None, run_name: Optional[str]=None, nested: bool=False, tags: Optional[Dict[str, Any]]=None, remove_config_values: List[str]=SimpleFrozenList()):
    config = nlp.config.interpolate()
    config_dot = dict_to_dot(config)
    for field in remove_config_values:
        del config_dot[field]
    config = dot_to_dict(config_dot)
    mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=run_name, nested=nested, tags=tags)
    config_dot_items = list(config_dot.items())
    config_dot_batches = [config_dot_items[i:i + 100] for i in range(0, len(config_dot_items), 100)]
    for batch in config_dot_batches:
        mlflow.log_params({k.replace('@', ''): v for k, v in batch})