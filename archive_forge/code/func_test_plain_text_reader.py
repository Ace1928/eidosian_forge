import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Generator, Iterable, List, TextIO, Tuple
import pytest
from spacy.lang.en import English
from spacy.training import Example, PlainTextCorpus
from spacy.util import make_tempdir
@pytest.mark.parametrize('min_length', [0, 5])
@pytest.mark.parametrize('max_length', [0, 5])
def test_plain_text_reader(min_length, max_length):
    nlp = English()
    with _string_to_tmp_file(PLAIN_TEXT_DOC) as file_path:
        corpus = PlainTextCorpus(file_path, min_length=min_length, max_length=max_length)
        check = [doc for doc in PLAIN_TEXT_DOC_TOKENIZED if len(doc) >= min_length and (max_length == 0 or len(doc) <= max_length)]
        reference, predicted = _examples_to_tokens(corpus(nlp))
        assert reference == check
        assert predicted == check