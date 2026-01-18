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
def tokenizer_factory(nlp: 'Language') -> Tokenizer:
    prefixes = nlp.Defaults.prefixes
    suffixes = nlp.Defaults.suffixes
    infixes = nlp.Defaults.infixes
    prefix_search = util.compile_prefix_regex(prefixes).search if prefixes else None
    suffix_search = util.compile_suffix_regex(suffixes).search if suffixes else None
    infix_finditer = util.compile_infix_regex(infixes).finditer if infixes else None
    return Tokenizer(nlp.vocab, rules=nlp.Defaults.tokenizer_exceptions, prefix_search=prefix_search, suffix_search=suffix_search, infix_finditer=infix_finditer, token_match=nlp.Defaults.token_match, url_match=nlp.Defaults.url_match)