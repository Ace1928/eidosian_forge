from spacy.tokens import Doc
def test_tr_noun_chunks_flat_simple(tr_tokenizer):
    text = 'New York'
    heads = [0, 0]
    deps = ['ROOT', 'flat']
    pos = ['PROPN', 'PROPN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'New York '