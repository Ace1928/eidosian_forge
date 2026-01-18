from typing import List
import pytest
import spacy
from spacy.training import Example
@pytest.mark.parametrize('component', ['ner', 'tagger', 'parser', 'textcat_multilabel'])
def test_rehearse(component):
    nlp = spacy.blank('en')
    nlp.add_pipe(component)
    nlp = _optimize(nlp, component, TRAIN_DATA, False)
    _optimize(nlp, component, REHEARSE_DATA, True)