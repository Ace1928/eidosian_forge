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
def test_update_with_annotates():
    name = 'test_with_annotates'
    results = {}

    def make_component(name):
        results[name] = ''

        def component(doc):
            nonlocal results
            results[name] += doc.text
            return doc
        return component
    Language.component(f'{name}1', func=make_component(f'{name}1'))
    Language.component(f'{name}2', func=make_component(f'{name}2'))
    components = set([f'{name}1', f'{name}2'])
    nlp = English()
    texts = ['a', 'bb', 'ccc']
    examples = []
    for text in texts:
        examples.append(Example(nlp.make_doc(text), nlp.make_doc(text)))
    for components_to_annotate in [[], [f'{name}1'], [f'{name}1', f'{name}2'], [f'{name}2', f'{name}1']]:
        for key in results:
            results[key] = ''
        nlp = English(vocab=nlp.vocab)
        nlp.add_pipe(f'{name}1')
        nlp.add_pipe(f'{name}2')
        nlp.update(examples, annotates=components_to_annotate)
        for component in components_to_annotate:
            assert results[component] == ''.join((eg.predicted.text for eg in examples))
        for component in components - set(components_to_annotate):
            assert results[component] == ''