import logging
import unittest
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.topic_coherence import indirect_confirmation_measure
from gensim.topic_coherence import text_analysis
Sanity check word2vec_similarity.