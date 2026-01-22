from typing import Dict, Any, Tuple, Callable, List, Optional, IO
from types import ModuleType
import os
import sys
from spacy import Language
from spacy.util import SimpleFrozenList
from .util import dict_to_dot, dot_to_dict, matcher_for_regex_patterns
from .util import setup_default_console_logger, LoggerT
def _setup_clearml(clearml: ModuleType, nlp: 'Language', project_name: str, task_name: str, log_dataset_dir: Optional[str]=None, log_best_dir: Optional[str]=None, log_latest_dir: Optional[str]=None, remove_config_values: List[str]=SimpleFrozenList()) -> Tuple[Any, Any, Any]:
    config = nlp.config.interpolate()
    config_dot = dict_to_dot(config)
    for field in remove_config_values:
        del config_dot[field]
    config = dot_to_dict(config_dot)
    task = clearml.Task.init(project_name=project_name, task_name=task_name, output_uri=True)
    for config_section, subconfig_or_value in config.items():
        task.connect(subconfig_or_value, name=config_section)
    if log_dataset_dir:
        dataset = clearml.Dataset.create(dataset_project=project_name, dataset_name=os.path.basename(log_dataset_dir))
        dataset.add_files(log_dataset_dir)
        dataset.finalize(auto_upload=True)
        task.set_user_properties({'name': 'Created Dataset ID', 'value': dataset.id})
    if log_best_dir:
        best_model = clearml.OutputModel(task=task, framework='spaCy', name='Best Model')
    else:
        best_model = None
    if log_latest_dir:
        last_model = clearml.OutputModel(task=task, framework='spaCy', name='Last Model')
    else:
        last_model = None
    return (task, best_model, last_model)