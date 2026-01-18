import pytest
from spacy.lang.en import English
from spacy.training import Example
from thinc.api import Config
@pytest.mark.parametrize('name,train_data,textcat_config', [('textcat', TRAIN_DATA_SINGLE_LABEL, {'@architectures': 'spacy-legacy.TextCatCNN.v1', 'exclusive_classes': True, 'tok2vec': DEFAULT_TOK2VEC_MODEL}), ('textcat_multilabel', TRAIN_DATA_MULTI_LABEL, {'@architectures': 'spacy-legacy.TextCatCNN.v1', 'exclusive_classes': False, 'tok2vec': DEFAULT_TOK2VEC_MODEL}), ('textcat', TRAIN_DATA_SINGLE_LABEL, {'@architectures': 'spacy-legacy.TextCatBOW.v1', 'exclusive_classes': False, 'ngram_size': 2, 'no_output_layer': True}), ('textcat_multilabel', TRAIN_DATA_MULTI_LABEL, {'@architectures': 'spacy-legacy.TextCatBOW.v1', 'exclusive_classes': True, 'ngram_size': 3, 'no_output_layer': True}), ('textcat', TRAIN_DATA_SINGLE_LABEL, {'@architectures': 'spacy-legacy.TextCatEnsemble.v1', 'width': 32, 'embed_size': 16, 'exclusive_classes': True, 'ngram_size': 2, 'window_size': 4, 'conv_depth': 1, 'pretrained_vectors': False, 'dropout': 0.1}), ('textcat_multilabel', TRAIN_DATA_MULTI_LABEL, {'@architectures': 'spacy-legacy.TextCatEnsemble.v1', 'width': 32, 'embed_size': 16, 'exclusive_classes': False, 'ngram_size': 3, 'window_size': 6, 'conv_depth': 2, 'pretrained_vectors': False, 'dropout': 0.2})])
def test_textcat(name, train_data, textcat_config):
    pipe_config = {'model': textcat_config}
    nlp = English()
    textcat = nlp.add_pipe(name, config=pipe_config)
    train_examples = []
    for text, annotations in train_data:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
        for label, value in annotations.get('cats').items():
            textcat.add_label(label)
    optimizer = nlp.initialize()
    for i in range(5):
        losses = {}
        nlp.update(train_examples, sgd=optimizer, losses=losses)