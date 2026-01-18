import pytest
from spacy.tokens import Doc
from ..util import apply_transition_sequence
@pytest.mark.parametrize('text,length', [(['\n'], 1), (['\n', '\t', '\n\n', '\t'], 4)])
@pytest.mark.skip(reason='The step_through API was removed (but should be brought back)')
def test_parser_space_attachment_space(en_parser, text, length):
    doc = Doc(en_parser.vocab, words=text)
    assert len(doc) == length
    with en_parser.step_through(doc) as _:
        pass
    assert doc[0].is_space
    for token in doc:
        assert token.head.i == length - 1