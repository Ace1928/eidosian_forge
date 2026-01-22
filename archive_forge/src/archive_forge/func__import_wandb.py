from typing import Dict, Any, Tuple, Callable, List, IO, Optional
from types import ModuleType
import sys
from spacy import Language
from spacy.util import SimpleFrozenList
from .util import dict_to_dot, dot_to_dict, matcher_for_regex_patterns
from .util import setup_default_console_logger, LoggerT
def _import_wandb() -> ModuleType:
    try:
        import wandb
        from wandb import init, log, join
        return wandb
    except ImportError:
        raise ImportError("The 'wandb' library could not be found - did you install it? Alternatively, specify the 'ConsoleLogger' in the 'training.logger' config section, instead of the 'WandbLogger'.")