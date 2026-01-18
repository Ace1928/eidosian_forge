from spacy.tokens import Doc
def test_tr_noun_chunks_two_adjs_simple(tr_tokenizer):
    text = 'beyaz tombik kedi'
    heads = [2, 2, 2]
    deps = ['amod', 'amod', 'ROOT']
    pos = ['ADJ', 'ADJ', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'beyaz tombik kedi '