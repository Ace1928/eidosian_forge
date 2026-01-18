from spacy.tokens import Doc
def test_tr_noun_chunks_nmod_amod(tr_tokenizer):
    text = 'okulun eski müdürü'
    heads = [2, 2, 2]
    deps = ['nmod', 'amod', 'ROOT']
    pos = ['NOUN', 'ADJ', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'okulun eski müdürü '