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
def _apply_pipes(ensure_doc: Callable[[Union[str, Doc, bytes], _AnyContext], Doc], pipes: Iterable[Callable[..., Iterator[Doc]]], receiver, sender, underscore_state: Tuple[dict, dict, dict]) -> None:
    """Worker for Language.pipe

    ensure_doc (Callable[[Union[str, Doc]], Doc]): Function to create Doc from text
        or raise an error if the input is neither a Doc nor a string.
    pipes (Iterable[Pipe]): The components to apply.
    receiver (multiprocessing.Connection): Pipe to receive text. Usually
        created by `multiprocessing.Pipe()`
    sender (multiprocessing.Connection): Pipe to send doc. Usually created by
        `multiprocessing.Pipe()`
    underscore_state (Tuple[dict, dict, dict]): The data in the Underscore class
        of the parent.
    """
    Underscore.load_state(underscore_state)
    while True:
        try:
            texts_with_ctx = receiver.get()
            if isinstance(texts_with_ctx, _WorkDoneSentinel):
                sender.close()
                receiver.close()
                return
            docs = (ensure_doc(doc_like, context) for doc_like, context in texts_with_ctx)
            for pipe in pipes:
                docs = pipe(docs)
            byte_docs = [(doc.to_bytes(), doc._context, None) for doc in docs]
            padding = [(None, None, None)] * (len(texts_with_ctx) - len(byte_docs))
            data: Sequence[Tuple[Optional[bytes], Optional[Any], Optional[bytes]]] = byte_docs + padding
        except Exception:
            error_msg = [(None, None, srsly.msgpack_dumps(traceback.format_exc()))]
            padding = [(None, None, None)] * (len(texts_with_ctx) - 1)
            data = error_msg + padding
        try:
            sender.send(data)
        except BrokenPipeError:
            sender.close()
            receiver.close()
            return