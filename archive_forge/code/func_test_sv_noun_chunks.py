import pytest
from spacy.tokens import Doc
@pytest.mark.parametrize('text,pos,deps,heads,expected_noun_chunks', SV_NP_TEST_EXAMPLES)
def test_sv_noun_chunks(sv_tokenizer, text, pos, deps, heads, expected_noun_chunks):
    tokens = sv_tokenizer(text)
    assert len(heads) == len(pos)
    words = [t.text for t in tokens]
    doc = Doc(tokens.vocab, words=words, heads=heads, deps=deps, pos=pos)
    noun_chunks = list(doc.noun_chunks)
    assert len(noun_chunks) == len(expected_noun_chunks)
    for i, np in enumerate(noun_chunks):
        assert np.text == expected_noun_chunks[i]