from spacy.tokens import Doc
def test_tr_noun_chunks_chain_nmod_with_adj(tr_tokenizer):
    text = 'ev sahibinin tatlı köpeği'
    heads = [1, 3, 3, 3]
    deps = ['nmod', 'nmod', 'amod', 'ROOT']
    pos = ['NOUN', 'NOUN', 'ADJ', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'ev sahibinin tatlı köpeği '