from spacy.tokens import Doc
def test_tr_noun_chunks_nmod_two(tr_tokenizer):
    text = 'kızın saçının rengi'
    heads = [1, 2, 2]
    deps = ['nmod', 'nmod', 'ROOT']
    pos = ['NOUN', 'NOUN', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'kızın saçının rengi '