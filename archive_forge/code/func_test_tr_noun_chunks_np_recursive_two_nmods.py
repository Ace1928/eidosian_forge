from spacy.tokens import Doc
def test_tr_noun_chunks_np_recursive_two_nmods(tr_tokenizer):
    text = 'ustanın kapısını degiştireceği çamasır makinası'
    heads = [2, 2, 4, 4, 4]
    deps = ['nsubj', 'obj', 'acl', 'nmod', 'ROOT']
    pos = ['NOUN', 'NOUN', 'VERB', 'NOUN', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'ustanın kapısını degiştireceği çamasır makinası '