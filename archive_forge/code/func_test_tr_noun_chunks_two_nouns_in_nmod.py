from spacy.tokens import Doc
def test_tr_noun_chunks_two_nouns_in_nmod(tr_tokenizer):
    text = 'kız ve erkek çocuklar'
    heads = [3, 2, 0, 3]
    deps = ['nmod', 'cc', 'conj', 'ROOT']
    pos = ['NOUN', 'CCONJ', 'NOUN', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'kız ve erkek çocuklar '