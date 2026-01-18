import pytest
from spacy.tokens import Doc
def test_noun_chunks_is_parsed_it(it_tokenizer):
    """Test that noun_chunks raises Value Error for 'it' language if Doc is not parsed."""
    doc = it_tokenizer('Sei andato a Oxford')
    with pytest.raises(ValueError):
        list(doc.noun_chunks)