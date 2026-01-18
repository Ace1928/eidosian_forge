from spacy.tokens import Doc
def test_tr_noun_chunks_acl_nmod2(tr_tokenizer):
    text = 'bildiğim bir turizm şirketi'
    heads = [3, 3, 3, 3]
    deps = ['acl', 'det', 'nmod', 'ROOT']
    pos = ['VERB', 'DET', 'NOUN', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'bildiğim bir turizm şirketi '