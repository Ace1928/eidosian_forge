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
def _get_pipe_index(self, before: Optional[Union[str, int]]=None, after: Optional[Union[str, int]]=None, first: Optional[bool]=None, last: Optional[bool]=None) -> int:
    """Determine where to insert a pipeline component based on the before/
        after/first/last values.

        before (str): Name or index of the component to insert directly before.
        after (str): Name or index of component to insert directly after.
        first (bool): If True, insert component first in the pipeline.
        last (bool): If True, insert component last in the pipeline.
        RETURNS (int): The index of the new pipeline component.
        """
    all_args = {'before': before, 'after': after, 'first': first, 'last': last}
    if sum((arg is not None for arg in [before, after, first, last])) >= 2:
        raise ValueError(Errors.E006.format(args=all_args, opts=self.component_names))
    if last or not any((value is not None for value in [first, before, after])):
        return len(self._components)
    elif first:
        return 0
    elif isinstance(before, str):
        if before not in self.component_names:
            raise ValueError(Errors.E001.format(name=before, opts=self.component_names))
        return self.component_names.index(before)
    elif isinstance(after, str):
        if after not in self.component_names:
            raise ValueError(Errors.E001.format(name=after, opts=self.component_names))
        return self.component_names.index(after) + 1
    elif type(before) == int:
        if before >= len(self._components) or before < 0:
            err = Errors.E959.format(dir='before', idx=before, opts=self.component_names)
            raise ValueError(err)
        return before
    elif type(after) == int:
        if after >= len(self._components) or after < 0:
            err = Errors.E959.format(dir='after', idx=after, opts=self.component_names)
            raise ValueError(err)
        return after + 1
    raise ValueError(Errors.E006.format(args=all_args, opts=self.component_names))