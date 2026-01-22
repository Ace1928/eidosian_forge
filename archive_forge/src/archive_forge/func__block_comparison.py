import math
import re
from nltk.tokenize.api import TokenizerI
def _block_comparison(self, tokseqs, token_table):
    """Implements the block comparison method"""

    def blk_frq(tok, block):
        ts_occs = filter(lambda o: o[0] in block, token_table[tok].ts_occurences)
        freq = sum((tsocc[1] for tsocc in ts_occs))
        return freq
    gap_scores = []
    numgaps = len(tokseqs) - 1
    for curr_gap in range(numgaps):
        score_dividend, score_divisor_b1, score_divisor_b2 = (0.0, 0.0, 0.0)
        score = 0.0
        if curr_gap < self.k - 1:
            window_size = curr_gap + 1
        elif curr_gap > numgaps - self.k:
            window_size = numgaps - curr_gap
        else:
            window_size = self.k
        b1 = [ts.index for ts in tokseqs[curr_gap - window_size + 1:curr_gap + 1]]
        b2 = [ts.index for ts in tokseqs[curr_gap + 1:curr_gap + window_size + 1]]
        for t in token_table:
            score_dividend += blk_frq(t, b1) * blk_frq(t, b2)
            score_divisor_b1 += blk_frq(t, b1) ** 2
            score_divisor_b2 += blk_frq(t, b2) ** 2
        try:
            score = score_dividend / math.sqrt(score_divisor_b1 * score_divisor_b2)
        except ZeroDivisionError:
            pass
        gap_scores.append(score)
    return gap_scores