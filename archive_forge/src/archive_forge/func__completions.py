from __future__ import annotations
import builtins as builtin_mod
import enum
import glob
import inspect
import itertools
import keyword
import os
import re
import string
import sys
import tokenize
import time
import unicodedata
import uuid
import warnings
from ast import literal_eval
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
from types import SimpleNamespace
from typing import (
from IPython.core.guarded_eval import guarded_eval, EvaluationContext
from IPython.core.error import TryNext
from IPython.core.inputtransformer2 import ESC_MAGIC
from IPython.core.latex_symbols import latex_symbols, reverse_latex_symbol
from IPython.core.oinspect import InspectColors
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import generics
from IPython.utils.decorators import sphinx_options
from IPython.utils.dir2 import dir2, get_real_method
from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.path import ensure_dir_exists
from IPython.utils.process import arg_split
from traitlets import (
from traitlets.config.configurable import Configurable
import __main__
def _completions(self, full_text: str, offset: int, *, _timeout) -> Iterator[Completion]:
    """
        Core completion module.Same signature as :any:`completions`, with the
        extra `timeout` parameter (in seconds).

        Computing jedi's completion ``.type`` can be quite expensive (it is a
        lazy property) and can require some warm-up, more warm up than just
        computing the ``name`` of a completion. The warm-up can be :

            - Long warm-up the first time a module is encountered after
            install/update: actually build parse/inference tree.

            - first time the module is encountered in a session: load tree from
            disk.

        We don't want to block completions for tens of seconds so we give the
        completer a "budget" of ``_timeout`` seconds per invocation to compute
        completions types, the completions that have not yet been computed will
        be marked as "unknown" an will have a chance to be computed next round
        are things get cached.

        Keep in mind that Jedi is not the only thing treating the completion so
        keep the timeout short-ish as if we take more than 0.3 second we still
        have lots of processing to do.

        """
    deadline = time.monotonic() + _timeout
    before = full_text[:offset]
    cursor_line, cursor_column = position_to_cursor(full_text, offset)
    jedi_matcher_id = _get_matcher_id(self._jedi_matcher)

    def is_non_jedi_result(result: MatcherResult, identifier: str) -> TypeGuard[SimpleMatcherResult]:
        return identifier != jedi_matcher_id
    results = self._complete(full_text=full_text, cursor_line=cursor_line, cursor_pos=cursor_column)
    non_jedi_results: Dict[str, SimpleMatcherResult] = {identifier: result for identifier, result in results.items() if is_non_jedi_result(result, identifier)}
    jedi_matches = cast(_JediMatcherResult, results[jedi_matcher_id])['completions'] if jedi_matcher_id in results else ()
    iter_jm = iter(jedi_matches)
    if _timeout:
        for jm in iter_jm:
            try:
                type_ = jm.type
            except Exception:
                if self.debug:
                    print('Error in Jedi getting type of ', jm)
                type_ = None
            delta = len(jm.name_with_symbols) - len(jm.complete)
            if type_ == 'function':
                signature = _make_signature(jm)
            else:
                signature = ''
            yield Completion(start=offset - delta, end=offset, text=jm.name_with_symbols, type=type_, signature=signature, _origin='jedi')
            if time.monotonic() > deadline:
                break
    for jm in iter_jm:
        delta = len(jm.name_with_symbols) - len(jm.complete)
        yield Completion(start=offset - delta, end=offset, text=jm.name_with_symbols, type=_UNKNOWN_TYPE, _origin='jedi', signature='')
    if jedi_matches and non_jedi_results and self.debug:
        some_start_offset = before.rfind(next(iter(non_jedi_results.values()))['matched_fragment'])
        yield Completion(start=some_start_offset, end=offset, text='--jedi/ipython--', _origin='debug', type='none', signature='')
    ordered: List[Completion] = []
    sortable: List[Completion] = []
    for origin, result in non_jedi_results.items():
        matched_text = result['matched_fragment']
        start_offset = before.rfind(matched_text)
        is_ordered = result.get('ordered', False)
        container = ordered if is_ordered else sortable
        assert before.endswith(matched_text)
        for simple_completion in result['completions']:
            completion = Completion(start=start_offset, end=offset, text=simple_completion.text, _origin=origin, signature='', type=simple_completion.type or _UNKNOWN_TYPE)
            container.append(completion)
    yield from list(self._deduplicate(ordered + self._sort(sortable)))[:MATCHES_LIMIT]