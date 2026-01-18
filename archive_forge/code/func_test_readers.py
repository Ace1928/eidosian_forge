from typing import Callable, Dict, Iterable
import pytest
from thinc.api import Config, fix_random_seed
from spacy import Language
from spacy.schemas import ConfigSchemaTraining
from spacy.training import Example
from spacy.util import load_model_from_config, registry, resolve_dot_names
def test_readers():
    config_string = '\n    [training]\n\n    [corpora]\n    @readers = "myreader.v1"\n\n    [nlp]\n    lang = "en"\n    pipeline = ["tok2vec", "textcat"]\n\n    [components]\n\n    [components.tok2vec]\n    factory = "tok2vec"\n\n    [components.textcat]\n    factory = "textcat"\n    '

    @registry.readers('myreader.v1')
    def myreader() -> Dict[str, Callable[[Language], Iterable[Example]]]:
        annots = {'cats': {'POS': 1.0, 'NEG': 0.0}}

        def reader(nlp: Language):
            doc = nlp.make_doc(f'This is an example')
            return [Example.from_dict(doc, annots)]
        return {'train': reader, 'dev': reader, 'extra': reader, 'something': reader}
    config = Config().from_str(config_string)
    nlp = load_model_from_config(config, auto_fill=True)
    T = registry.resolve(nlp.config.interpolate()['training'], schema=ConfigSchemaTraining)
    dot_names = [T['train_corpus'], T['dev_corpus']]
    train_corpus, dev_corpus = resolve_dot_names(nlp.config, dot_names)
    assert isinstance(train_corpus, Callable)
    optimizer = T['optimizer']
    nlp.initialize(lambda: train_corpus(nlp), sgd=optimizer)
    for example in train_corpus(nlp):
        nlp.update([example], sgd=optimizer)
    scores = nlp.evaluate(list(dev_corpus(nlp)))
    assert scores['cats_macro_auc'] == 0.0
    doc = nlp('Quick test')
    assert doc.cats
    corpora = {'corpora': nlp.config.interpolate()['corpora']}
    extra_corpus = registry.resolve(corpora)['corpora']['extra']
    assert isinstance(extra_corpus, Callable)