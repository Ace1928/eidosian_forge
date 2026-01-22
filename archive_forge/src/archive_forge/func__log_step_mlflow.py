from typing import Dict, Any, Tuple, Callable, List, Optional, IO
from types import ModuleType
import os
import sys
from spacy import Language, load
from spacy.util import SimpleFrozenList
from .util import dict_to_dot, dot_to_dict, matcher_for_regex_patterns
from .util import setup_default_console_logger, LoggerT
def _log_step_mlflow(mlflow: ModuleType, info: Optional[Dict[str, Any]]):
    if info is None:
        return
    score = info['score']
    other_scores = info['other_scores']
    losses = info['losses']
    output_path = info.get('output_path', None)
    if score is not None:
        mlflow.log_metric('score', score)
    if losses:
        mlflow.log_metrics({f'loss_{k}': v for k, v in losses.items()})
    if isinstance(other_scores, dict):
        mlflow.log_metrics({k: v for k, v in dict_to_dot(other_scores).items() if isinstance(v, float) or isinstance(v, int)})
    if output_path and score == max(info['checkpoints'])[0]:
        nlp = load(output_path)
        mlflow.spacy.log_model(nlp, 'best')