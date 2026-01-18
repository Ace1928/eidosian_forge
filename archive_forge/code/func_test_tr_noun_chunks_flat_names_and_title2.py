from spacy.tokens import Doc
def test_tr_noun_chunks_flat_names_and_title2(tr_tokenizer):
    text = 'Ahmet Vefik Paşa'
    heads = [2, 0, 2]
    deps = ['nmod', 'flat', 'ROOT']
    pos = ['PROPN', 'PROPN', 'PROPN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'Ahmet Vefik Paşa '