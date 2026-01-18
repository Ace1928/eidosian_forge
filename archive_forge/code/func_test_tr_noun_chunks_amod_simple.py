from spacy.tokens import Doc
def test_tr_noun_chunks_amod_simple(tr_tokenizer):
    text = 'sarı kedi'
    heads = [1, 1]
    deps = ['amod', 'ROOT']
    pos = ['ADJ', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'sarı kedi '