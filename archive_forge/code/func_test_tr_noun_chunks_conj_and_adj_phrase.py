from spacy.tokens import Doc
def test_tr_noun_chunks_conj_and_adj_phrase(tr_tokenizer):
    text = 'ben ve akıllı çocuk'
    heads = [0, 3, 3, 0]
    deps = ['ROOT', 'cc', 'amod', 'conj']
    pos = ['PRON', 'CCONJ', 'ADJ', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 2
    assert chunks[0].text_with_ws == 'akıllı çocuk '
    assert chunks[1].text_with_ws == 'ben '