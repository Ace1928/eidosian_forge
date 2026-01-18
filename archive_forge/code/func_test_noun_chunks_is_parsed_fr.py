import pytest
from spacy.tokens import Doc
def test_noun_chunks_is_parsed_fr(fr_tokenizer):
    """Test that noun_chunks raises Value Error for 'fr' language if Doc is not parsed."""
    doc = fr_tokenizer("Je suis allé à l'école")
    with pytest.raises(ValueError):
        list(doc.noun_chunks)