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
def test_pipe_methods_frozen():
    """Test that spaCy raises custom error messages if "frozen" properties are
    accessed. We still want to use a list here to not break backwards
    compatibility, but users should see an error if they're trying to append
    to nlp.pipeline etc."""
    nlp = Language()
    ner = nlp.add_pipe('ner')
    assert nlp.pipe_names == ['ner']
    for prop in [nlp.pipeline, nlp.pipe_names, nlp.components, nlp.component_names, nlp.disabled, nlp.factory_names]:
        assert isinstance(prop, list)
        assert isinstance(prop, SimpleFrozenList)
    with pytest.raises(NotImplementedError):
        nlp.pipeline.append(('ner2', ner))
    with pytest.raises(NotImplementedError):
        nlp.pipe_names.pop()
    with pytest.raises(NotImplementedError):
        nlp.components.sort()
    with pytest.raises(NotImplementedError):
        nlp.component_names.clear()