import pytest
from thinc.api import ConfigValidationError, Linear, Model
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.tokens import Doc
from spacy.util import SimpleFrozenDict, combine_score_weights, registry
from ..util import make_tempdir
def test_pipe_factories_config_excludes_nlp():
    """Test that the extra values we temporarily add to component config
    blocks/functions are removed and not copied around.
    """
    name = 'test_pipe_factories_config_excludes_nlp'
    func = lambda nlp, name: lambda doc: doc
    Language.factory(name, func=func)
    config = {'nlp': {'lang': 'en', 'pipeline': [name]}, 'components': {name: {'factory': name}}}
    nlp = English.from_config(config)
    assert nlp.pipe_names == [name]
    pipe_cfg = nlp.get_pipe_config(name)
    pipe_cfg == {'factory': name}
    assert nlp._pipe_configs[name] == {'factory': name}