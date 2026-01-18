from pathlib import Path
from typing import Any, Callable, Dict, Iterable
import srsly
from numpy import zeros
from thinc.api import Config
from spacy import Errors, util
from spacy.kb.kb_in_memory import InMemoryLookupKB
from spacy.util import SimpleFrozenList, ensure_path, load_model_from_config, registry
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_serialize_kb_disk(en_vocab):
    kb1 = _get_dummy_kb(en_vocab)
    _check_kb(kb1)
    with make_tempdir() as d:
        dir_path = ensure_path(d)
        if not dir_path.exists():
            dir_path.mkdir()
        file_path = dir_path / 'kb'
        kb1.to_disk(str(file_path))
        kb2 = InMemoryLookupKB(vocab=en_vocab, entity_vector_length=3)
        kb2.from_disk(str(file_path))
    _check_kb(kb2)