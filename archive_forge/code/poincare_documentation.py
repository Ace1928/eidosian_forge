import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
Evaluate spearman scores for lexical entailment for given embedding.

        Parameters
        ----------
        embedding : :class:`~gensim.models.poincare.PoincareKeyedVectors`
            Embedding for which evaluation is to be done.

        Returns
        -------
        float
            Spearman correlation score for the task for input embedding.

        