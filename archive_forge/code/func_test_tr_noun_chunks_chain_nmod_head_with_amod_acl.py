from spacy.tokens import Doc
def test_tr_noun_chunks_chain_nmod_head_with_amod_acl(tr_tokenizer):
    text = 'arabanın kırdığım sol aynası'
    heads = [3, 3, 3, 3]
    deps = ['nmod', 'acl', 'amod', 'ROOT']
    pos = ['NOUN', 'VERB', 'ADJ', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'arabanın kırdığım sol aynası '