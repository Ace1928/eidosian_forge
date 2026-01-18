from spacy.tokens import Doc
def test_tr_noun_chunks_chain_nmod_with_acl(tr_tokenizer):
    text = 'ev sahibinin gelen köpeği'
    heads = [1, 3, 3, 3]
    deps = ['nmod', 'nmod', 'acl', 'ROOT']
    pos = ['NOUN', 'NOUN', 'VERB', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'ev sahibinin gelen köpeği '