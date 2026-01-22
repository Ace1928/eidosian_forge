import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise
def _sift_sentiment_scores(self, sentiments):
    pos_sum = 0.0
    neg_sum = 0.0
    neu_count = 0
    for sentiment_score in sentiments:
        if sentiment_score > 0:
            pos_sum += float(sentiment_score) + 1
        if sentiment_score < 0:
            neg_sum += float(sentiment_score) - 1
        if sentiment_score == 0:
            neu_count += 1
    return (pos_sum, neg_sum, neu_count)