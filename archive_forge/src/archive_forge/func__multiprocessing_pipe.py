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
def _multiprocessing_pipe(self, texts: Iterable[Union[str, Doc]], pipes: Iterable[Callable[..., Iterator[Doc]]], n_process: int, batch_size: int) -> Iterator[Doc]:

    def prepare_input(texts: Iterable[Union[str, Doc]]) -> Iterable[Tuple[Union[str, bytes], _AnyContext]]:
        for doc_like in texts:
            if isinstance(doc_like, Doc):
                yield (doc_like.to_bytes(), cast(_AnyContext, doc_like._context))
            else:
                yield (doc_like, cast(_AnyContext, None))
    serialized_texts_with_ctx = prepare_input(texts)
    texts, raw_texts = itertools.tee(serialized_texts_with_ctx)
    texts_q: List[mp.Queue] = [mp.Queue() for _ in range(n_process)]
    bytedocs_recv_ch, bytedocs_send_ch = zip(*[mp.Pipe(False) for _ in range(n_process)])
    batch_texts = util.minibatch(texts, batch_size)
    sender = _Sender(batch_texts, texts_q, chunk_size=n_process)
    sender.send()
    sender.send()
    procs = [mp.Process(target=_apply_pipes, args=(self._ensure_doc_with_context, pipes, rch, sch, Underscore.get_state())) for rch, sch in zip(texts_q, bytedocs_send_ch)]
    for proc in procs:
        proc.start()
    for tx in bytedocs_send_ch:
        tx.close()
    byte_tuples = chain.from_iterable((recv.recv() for recv in cycle(bytedocs_recv_ch)))
    try:
        for i, (_, (byte_doc, context, byte_error)) in enumerate(zip(raw_texts, byte_tuples), 1):
            if byte_doc is not None:
                doc = Doc(self.vocab).from_bytes(byte_doc)
                doc._context = context
                yield doc
            elif byte_error is not None:
                error = srsly.msgpack_loads(byte_error)
                self.default_error_handler(None, None, None, ValueError(Errors.E871.format(error=error)))
            if i % batch_size == 0:
                sender.step()
    finally:
        for q in texts_q:
            q.put(_WORK_DONE_SENTINEL)
            q.close()
        for r in bytedocs_recv_ch:
            r.close()
        for proc in procs:
            proc.join()
        if not all((proc.exitcode == 0 for proc in procs)):
            warnings.warn(Warnings.W127)