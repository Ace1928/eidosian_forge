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
class LinkPredictionEvaluation:
    """Evaluate reconstruction on given network for given embedding."""

    def __init__(self, train_path, test_path, embedding):
        """Initialize evaluation instance with tsv file containing relation pairs and embedding to be evaluated.

        Parameters
        ----------
        train_path : str
            Path to tsv file containing relation pairs used for training.
        test_path : str
            Path to tsv file containing relation pairs to evaluate.
        embedding : :class:`~gensim.models.poincare.PoincareKeyedVectors`
            Embedding to be evaluated.

        """
        items = set()
        relations = {'known': defaultdict(set), 'unknown': defaultdict(set)}
        data_files = {'known': train_path, 'unknown': test_path}
        for relation_type, data_file in data_files.items():
            with utils.open(data_file, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    assert len(row) == 2, 'Hypernym pair has more than two items'
                    item_1_index = embedding.get_index(row[0])
                    item_2_index = embedding.get_index(row[1])
                    relations[relation_type][item_1_index].add(item_2_index)
                    items.update([item_1_index, item_2_index])
        self.items = items
        self.relations = relations
        self.embedding = embedding

    @staticmethod
    def get_unknown_relation_ranks_and_avg_prec(all_distances, unknown_relations, known_relations):
        """Compute ranks and Average Precision of unknown positive relations.

        Parameters
        ----------
        all_distances : numpy.array of float
            Array of all distances for a specific item.
        unknown_relations : list of int
            List of indices of unknown positive relations.
        known_relations : list of int
            List of indices of known positive relations.

        Returns
        -------
        tuple (list of int, float)
            The list contains ranks of positive relations in the same order as `positive_relations`.
            The float is the Average Precision of the ranking, e.g. ([1, 2, 3, 20], 0.610).

        """
        unknown_relation_distances = all_distances[unknown_relations]
        negative_relation_distances = np.ma.array(all_distances, mask=False)
        negative_relation_distances.mask[unknown_relations] = True
        negative_relation_distances.mask[known_relations] = True
        ranks = (negative_relation_distances < unknown_relation_distances[:, np.newaxis]).sum(axis=1) + 1
        map_ranks = np.sort(ranks) + np.arange(len(ranks))
        avg_precision = (np.arange(1, len(map_ranks) + 1) / np.sort(map_ranks)).mean()
        return (list(ranks), avg_precision)

    def evaluate(self, max_n=None):
        """Evaluate all defined metrics for the link prediction task.

        Parameters
        ----------
        max_n : int, optional
            Maximum number of positive relations to evaluate, all if `max_n` is None.

        Returns
        -------
        dict of (str, float)
            (metric_name, metric_value) pairs, e.g. {'mean_rank': 50.3, 'MAP': 0.31}.

        """
        mean_rank, map_ = self.evaluate_mean_rank_and_map(max_n)
        return {'mean_rank': mean_rank, 'MAP': map_}

    def evaluate_mean_rank_and_map(self, max_n=None):
        """Evaluate mean rank and MAP for link prediction.

        Parameters
        ----------
        max_n : int, optional
            Maximum number of positive relations to evaluate, all if `max_n` is None.

        Returns
        -------
        tuple (float, float)
            (mean_rank, MAP), e.g (50.3, 0.31).

        """
        ranks = []
        avg_precision_scores = []
        for i, item in enumerate(self.items, start=1):
            if item not in self.relations['unknown']:
                continue
            unknown_relations = list(self.relations['unknown'][item])
            known_relations = list(self.relations['known'][item])
            item_term = self.embedding.index_to_key[item]
            item_distances = self.embedding.distances(item_term)
            unknown_relation_ranks, avg_precision = self.get_unknown_relation_ranks_and_avg_prec(item_distances, unknown_relations, known_relations)
            ranks += unknown_relation_ranks
            avg_precision_scores.append(avg_precision)
            if max_n is not None and i > max_n:
                break
        return (np.mean(ranks), np.mean(avg_precision_scores))