import functools
import inspect
import itertools
import multiprocessing as mp
import random
import traceback
import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain, cycle
from pathlib import Path
from timeit import default_timer as timer
from typing import (
import srsly
from thinc.api import Config, CupyOps, Optimizer, get_current_ops
from . import about, ty, util
from .compat import Literal
from .errors import Errors, Warnings
from .git_info import GIT_VERSION
from .lang.punctuation import TOKENIZER_INFIXES, TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES
from .lang.tokenizer_exceptions import BASE_EXCEPTIONS, URL_MATCH
from .lookups import load_lookups
from .pipe_analysis import analyze_pipes, print_pipe_analysis, validate_attrs
from .schemas import (
from .scorer import Scorer
from .tokenizer import Tokenizer
from .tokens import Doc
from .tokens.underscore import Underscore
from .training import Example, validate_examples
from .training.initialize import init_tok2vec, init_vocab
from .util import (
from .vectors import BaseVectors
from .vocab import Vocab, create_vocab
@dataclass
class FactoryMeta:
    """Dataclass containing information about a component and its defaults
    provided by the @Language.component or @Language.factory decorator. It's
    created whenever a component is defined and stored on the Language class for
    each component instance and factory instance.
    """
    factory: str
    default_config: Optional[Dict[str, Any]] = None
    assigns: Iterable[str] = tuple()
    requires: Iterable[str] = tuple()
    retokenizes: bool = False
    scores: Iterable[str] = tuple()
    default_score_weights: Optional[Dict[str, Optional[float]]] = None