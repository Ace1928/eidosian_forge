from typing import Any, Dict, Iterator
from ...language import BaseDefaults, Language
from ...scorer import Scorer
from ...symbols import POS, X
from ...tokens import Doc
from ...training import validate_examples
from ...util import DummyTokenizer, load_config_from_str, registry
from ...vocab import Vocab
from .lex_attrs import LEX_ATTRS
from .punctuation import TOKENIZER_INFIXES
from .stop_words import STOP_WORDS
from .tag_map import TAG_MAP
class KoreanDefaults(BaseDefaults):
    config = load_config_from_str(DEFAULT_CONFIG)
    lex_attr_getters = LEX_ATTRS
    stop_words = STOP_WORDS
    writing_system = {'direction': 'ltr', 'has_case': False, 'has_letters': False}
    infixes = TOKENIZER_INFIXES