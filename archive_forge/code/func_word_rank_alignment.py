import math
from itertools import islice
from nltk.util import choose, ngrams
def word_rank_alignment(reference, hypothesis, character_based=False):
    """
    This is the word rank alignment algorithm described in the paper to produce
    the *worder* list, i.e. a list of word indices of the hypothesis word orders
    w.r.t. the list of reference words.

    Below is (H0, R0) example from the Isozaki et al. 2010 paper,
    note the examples are indexed from 1 but the results here are indexed from 0:

        >>> ref = str('he was interested in world history because he '
        ... 'read the book').split()
        >>> hyp = str('he read the book because he was interested in world '
        ... 'history').split()
        >>> word_rank_alignment(ref, hyp)
        [7, 8, 9, 10, 6, 0, 1, 2, 3, 4, 5]

    The (H1, R1) example from the paper, note the 0th index:

        >>> ref = 'John hit Bob yesterday'.split()
        >>> hyp = 'Bob hit John yesterday'.split()
        >>> word_rank_alignment(ref, hyp)
        [2, 1, 0, 3]

    Here is the (H2, R2) example from the paper, note the 0th index here too:

        >>> ref = 'the boy read the book'.split()
        >>> hyp = 'the book was read by the boy'.split()
        >>> word_rank_alignment(ref, hyp)
        [3, 4, 2, 0, 1]

    :param reference: a reference sentence
    :type reference: list(str)
    :param hypothesis: a hypothesis sentence
    :type hypothesis: list(str)
    """
    worder = []
    hyp_len = len(hypothesis)
    ref_ngrams = []
    hyp_ngrams = []
    for n in range(1, len(reference) + 1):
        for ng in ngrams(reference, n):
            ref_ngrams.append(ng)
        for ng in ngrams(hypothesis, n):
            hyp_ngrams.append(ng)
    for i, h_word in enumerate(hypothesis):
        if h_word not in reference:
            continue
        elif hypothesis.count(h_word) == reference.count(h_word) == 1:
            worder.append(reference.index(h_word))
        else:
            max_window_size = max(i, hyp_len - i + 1)
            for window in range(1, max_window_size):
                if i + window < hyp_len:
                    right_context_ngram = tuple(islice(hypothesis, i, i + window + 1))
                    num_times_in_ref = ref_ngrams.count(right_context_ngram)
                    num_times_in_hyp = hyp_ngrams.count(right_context_ngram)
                    if num_times_in_ref == num_times_in_hyp == 1:
                        pos = position_of_ngram(right_context_ngram, reference)
                        worder.append(pos)
                        break
                if window <= i:
                    left_context_ngram = tuple(islice(hypothesis, i - window, i + 1))
                    num_times_in_ref = ref_ngrams.count(left_context_ngram)
                    num_times_in_hyp = hyp_ngrams.count(left_context_ngram)
                    if num_times_in_ref == num_times_in_hyp == 1:
                        pos = position_of_ngram(left_context_ngram, reference)
                        worder.append(pos + len(left_context_ngram) - 1)
                        break
    return worder