from spacy.tokens import Doc
def test_tr_noun_chunks_flat_in_nmod(tr_tokenizer):
    text = 'Ahmet Sezer adında bir ögrenci'
    heads = [2, 0, 4, 4, 4]
    deps = ['nmod', 'flat', 'nmod', 'det', 'ROOT']
    pos = ['PROPN', 'PROPN', 'NOUN', 'DET', 'NOUN']
    tokens = tr_tokenizer(text)
    doc = Doc(tokens.vocab, words=[t.text for t in tokens], pos=pos, heads=heads, deps=deps)
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 1
    assert chunks[0].text_with_ws == 'Ahmet Sezer adında bir ögrenci '