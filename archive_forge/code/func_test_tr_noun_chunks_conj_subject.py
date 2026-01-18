from spacy.tokens import Doc
def test_tr_noun_chunks_conj_subject(tr_tokenizer):
    text = 'Sen ve ben iyi anlaşıyoruz'
    heads = [4, 2, 0, 2, 4]
    deps = ['nsubj', 'cc', 'conj', 'adv', 'ROOT']
    pos = ['PRON', 'CCONJ', 'PRON', 'ADV', 'VERB']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 2
    assert chunks[0].text_with_ws == 'ben '
    assert chunks[1].text_with_ws == 'Sen '