from nltk.translate.ribes_score import corpus_ribes, word_rank_alignment
def test_ribes_empty_worder():
    hyp = 'This is a nice sentence which I quite like'.split()
    ref = "Okay well that's neat and all but the reference's different".split()
    assert word_rank_alignment(ref, hyp) == []
    list_of_refs = [[ref]]
    hypotheses = [hyp]
    assert corpus_ribes(list_of_refs, hypotheses) == 0.0