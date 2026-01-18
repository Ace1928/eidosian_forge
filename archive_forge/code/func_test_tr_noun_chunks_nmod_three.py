from spacy.tokens import Doc
def test_tr_noun_chunks_nmod_three(tr_tokenizer):
    text = 'güney Afrika ülkelerinden Mozambik'
    heads = [1, 2, 3, 3]
    deps = ['nmod', 'nmod', 'nmod', 'ROOT']
    pos = ['NOUN', 'PROPN', 'NOUN', 'PROPN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'güney Afrika ülkelerinden Mozambik '