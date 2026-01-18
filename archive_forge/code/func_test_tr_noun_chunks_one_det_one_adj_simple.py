from spacy.tokens import Doc
def test_tr_noun_chunks_one_det_one_adj_simple(tr_tokenizer):
    text = 'O sarı kedi'
    heads = [2, 2, 2]
    deps = ['det', 'amod', 'ROOT']
    pos = ['DET', 'ADJ', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'O sarı kedi '