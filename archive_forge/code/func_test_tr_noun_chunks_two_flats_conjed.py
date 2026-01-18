from spacy.tokens import Doc
def test_tr_noun_chunks_two_flats_conjed(tr_tokenizer):
    text = 'New York ve Sierra Leone'
    heads = [0, 0, 3, 0, 3]
    deps = ['ROOT', 'flat', 'cc', 'conj', 'flat']
    pos = ['PROPN', 'PROPN', 'CCONJ', 'PROPN', 'PROPN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 2
    assert chunks[0].text_with_ws == 'Sierra Leone '
    assert chunks[1].text_with_ws == 'New York '