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
@staticmethod
def vector_distance_batch(vector_1, vectors_all):
    """Compute poincare distances between one vector and a set of other vectors.

        Parameters
        ----------
        vector_1 : numpy.array
            vector from which Poincare distances are to be computed, expected shape (dim,).
        vectors_all : numpy.array
            for each row in vectors_all, distance from vector_1 is computed, expected shape (num_vectors, dim).

        Returns
        -------
        numpy.array
            Poincare distance between `vector_1` and each row in `vectors_all`, shape (num_vectors,).

        """
    euclidean_dists = np.linalg.norm(vector_1 - vectors_all, axis=1)
    norm = np.linalg.norm(vector_1)
    all_norms = np.linalg.norm(vectors_all, axis=1)
    return np.arccosh(1 + 2 * (euclidean_dists ** 2 / ((1 - norm ** 2) * (1 - all_norms ** 2))))