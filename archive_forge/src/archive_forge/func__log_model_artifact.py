from typing import Dict, Any, Tuple, Callable, List, IO, Optional
from types import ModuleType
import sys
from spacy import Language
from spacy.util import SimpleFrozenList
from .util import dict_to_dot, dot_to_dict, matcher_for_regex_patterns
from .util import setup_default_console_logger, LoggerT
def _log_model_artifact(wandb: ModuleType, info: Optional[Dict[str, Any]], run: Any, model_log_interval: Optional[int]=None):
    if info is not None:
        if model_log_interval and info.get('output_path'):
            if info['step'] % model_log_interval == 0 and info['step'] != 0:
                _log_dir_artifact(wandb, path=info['output_path'], name='pipeline_' + run.id, type='checkpoint', metadata=info, aliases=[f'epoch {info['epoch']} step {info['step']}', 'latest', 'best' if info['score'] == max(info['checkpoints'])[0] else ''])