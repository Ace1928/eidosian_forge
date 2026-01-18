import pytest
from thinc.api import Config
from spacy import util
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline.span_finder import span_finder_default_config
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import fix_random_seed, make_tempdir, registry
@pytest.mark.parametrize('min_length, max_length, span_count', [(0, 0, 0), (None, None, 8), (2, None, 6), (None, 1, 2), (2, 3, 2)])
def test_set_annotations_span_lengths(min_length, max_length, span_count):
    nlp = Language()
    doc = nlp('Me and Jenny goes together like peas and carrots.')
    if min_length == 0 and max_length == 0:
        with pytest.raises(ValueError, match="Both 'min_length' and 'max_length'"):
            span_finder = nlp.add_pipe('span_finder', config={'max_length': max_length, 'min_length': min_length, 'spans_key': SPANS_KEY})
        return
    span_finder = nlp.add_pipe('span_finder', config={'max_length': max_length, 'min_length': min_length, 'spans_key': SPANS_KEY})
    nlp.initialize()
    scores = [(1, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 1), (0, 0)]
    span_finder.set_annotations([doc], scores)
    assert doc.spans[SPANS_KEY]
    assert len(doc.spans[SPANS_KEY]) == span_count
    if max_length is None:
        max_length = float('inf')
    if min_length is None:
        min_length = 1
    assert all((min_length <= len(span) <= max_length for span in doc.spans[SPANS_KEY]))