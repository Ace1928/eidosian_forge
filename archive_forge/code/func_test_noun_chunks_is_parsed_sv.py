import pytest
from spacy.tokens import Doc
def test_noun_chunks_is_parsed_sv(sv_tokenizer):
    """Test that noun_chunks raises Value Error for 'sv' language if Doc is not parsed."""
    doc = sv_tokenizer('Studenten läste den bästa boken')
    with pytest.raises(ValueError):
        list(doc.noun_chunks)