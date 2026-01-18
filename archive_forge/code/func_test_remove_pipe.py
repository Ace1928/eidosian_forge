import gc
import numpy
import pytest
from thinc.api import get_current_ops
import spacy
from spacy.lang.en import English
from spacy.lang.en.syntax_iterators import noun_chunks
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import SimpleFrozenList, get_arg_names, make_tempdir
from spacy.vocab import Vocab
@pytest.mark.parametrize('name', ['my_component'])
def test_remove_pipe(nlp, name):
    with pytest.raises(ValueError):
        nlp.remove_pipe(name)
    nlp.add_pipe('new_pipe', name=name)
    assert len(nlp.pipeline) == 1
    removed_name, removed_component = nlp.remove_pipe(name)
    assert not len(nlp.pipeline)
    assert removed_name == name
    assert removed_component == new_pipe