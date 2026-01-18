import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise

        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.

        :note: Hashtags are not taken into consideration (e.g. #BAD is neutral). If you
            are interested in processing the text in the hashtags too, then we recommend
            preprocessing your data to remove the #, after which the hashtag text may be
            matched as if it was a normal word in the sentence.
        