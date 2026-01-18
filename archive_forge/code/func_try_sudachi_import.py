import re
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import srsly
from thinc.api import Model
from ... import util
from ...errors import Errors
from ...language import BaseDefaults, Language
from ...pipeline import Morphologizer
from ...pipeline.morphologizer import DEFAULT_MORPH_MODEL
from ...scorer import Scorer
from ...symbols import POS
from ...tokens import Doc, MorphAnalysis
from ...training import validate_examples
from ...util import DummyTokenizer, load_config_from_str, registry
from ...vocab import Vocab
from .stop_words import STOP_WORDS
from .syntax_iterators import SYNTAX_ITERATORS
from .tag_bigram_map import TAG_BIGRAM_MAP
from .tag_map import TAG_MAP
from .tag_orth_map import TAG_ORTH_MAP
def try_sudachi_import(split_mode='A'):
    """SudachiPy is required for Japanese support, so check for it.
    It it's not available blow up and explain how to fix it.
    split_mode should be one of these values: "A", "B", "C", None->"A"."""
    try:
        from sudachipy import dictionary, tokenizer
        split_mode = {None: tokenizer.Tokenizer.SplitMode.A, 'A': tokenizer.Tokenizer.SplitMode.A, 'B': tokenizer.Tokenizer.SplitMode.B, 'C': tokenizer.Tokenizer.SplitMode.C}[split_mode]
        tok = dictionary.Dictionary().create(mode=split_mode)
        return tok
    except ImportError:
        raise ImportError('Japanese support requires SudachiPy and SudachiDict-core (https://github.com/WorksApplications/SudachiPy). Install with `pip install sudachipy sudachidict_core` or install spaCy with `pip install spacy[ja]`.') from None