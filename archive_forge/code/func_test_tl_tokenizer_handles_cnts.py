import pytest
from spacy.lang.tl.lex_attrs import like_num
@pytest.mark.parametrize('text,length', [('Huwag mo nang itanong sa akin.', 7), ('Nasubukan mo na bang hulihin ang hangin?', 8), ('Hindi ba?', 3), ('Nagbukas ang DFA ng 1,000 appointment slots para sa pasaporte.', 11), ("'Wala raw pasok bukas kasi may bagyo!' sabi ni Micah.", 14), ("'Ingat,' aniya. 'Maingay sila pag malayo at tahimik kung malapit.'", 17)])
def test_tl_tokenizer_handles_cnts(tl_tokenizer, text, length):
    tokens = tl_tokenizer(text)
    assert len(tokens) == length