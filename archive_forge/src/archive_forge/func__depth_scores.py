import math
import re
from nltk.tokenize.api import TokenizerI
def _depth_scores(self, scores):
    """Calculates the depth of each gap, i.e. the average difference
        between the left and right peaks and the gap's score"""
    depth_scores = [0 for x in scores]
    clip = min(max(len(scores) // 10, 2), 5)
    index = clip
    for gapscore in scores[clip:-clip]:
        lpeak = gapscore
        for score in scores[index::-1]:
            if score >= lpeak:
                lpeak = score
            else:
                break
        rpeak = gapscore
        for score in scores[index:]:
            if score >= rpeak:
                rpeak = score
            else:
                break
        depth_scores[index] = lpeak + rpeak - 2 * gapscore
        index += 1
    return depth_scores