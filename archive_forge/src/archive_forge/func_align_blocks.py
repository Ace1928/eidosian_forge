import math
def align_blocks(source_sents_lens, target_sents_lens, params=LanguageIndependent):
    """Return the sentence alignment of two text blocks (usually paragraphs).

        >>> align_blocks([5,5,5], [7,7,7])
        [(0, 0), (1, 1), (2, 2)]
        >>> align_blocks([10,5,5], [12,20])
        [(0, 0), (1, 1), (2, 1)]
        >>> align_blocks([12,20], [10,5,5])
        [(0, 0), (1, 1), (1, 2)]
        >>> align_blocks([10,2,10,10,2,10], [12,3,20,3,12])
        [(0, 0), (1, 1), (2, 2), (3, 2), (4, 3), (5, 4)]

    @param source_sents_lens: The list of source sentence lengths.
    @param target_sents_lens: The list of target sentence lengths.
    @param params: the sentence alignment parameters.
    @return: The sentence alignments, a list of index pairs.
    """
    alignment_types = list(params.PRIORS.keys())
    D = [[]]
    backlinks = {}
    for i in range(len(source_sents_lens) + 1):
        for j in range(len(target_sents_lens) + 1):
            min_dist = float('inf')
            min_align = None
            for a in alignment_types:
                prev_i = -1 - a[0]
                prev_j = j - a[1]
                if prev_i < -len(D) or prev_j < 0:
                    continue
                p = D[prev_i][prev_j] + align_log_prob(i, j, source_sents_lens, target_sents_lens, a, params)
                if p < min_dist:
                    min_dist = p
                    min_align = a
            if min_dist == float('inf'):
                min_dist = 0
            backlinks[i, j] = min_align
            D[-1].append(min_dist)
        if len(D) > 2:
            D.pop(0)
        D.append([])
    return trace(backlinks, source_sents_lens, target_sents_lens)