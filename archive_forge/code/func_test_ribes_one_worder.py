from nltk.translate.ribes_score import corpus_ribes, word_rank_alignment
def test_ribes_one_worder():
    hyp = 'This is a nice sentence which I quite like'.split()
    ref = "Okay well that's nice and all but the reference's different".split()
    assert word_rank_alignment(ref, hyp) == [3]
    list_of_refs = [[ref]]
    hypotheses = [hyp]
    assert corpus_ribes(list_of_refs, hypotheses) == 0.0