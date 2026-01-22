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
class PoincareBatch:
    """Compute Poincare distances, gradients and loss for a training batch.

    Store intermediate state to avoid recomputing multiple times.

    """

    def __init__(self, vectors_u, vectors_v, indices_u, indices_v, regularization_coeff=1.0):
        """
        Initialize instance with sets of vectors for which distances are to be computed.

        Parameters
        ----------
        vectors_u : numpy.array
            Vectors of all nodes `u` in the batch. Expected shape (batch_size, dim).
        vectors_v : numpy.array
            Vectors of all positively related nodes `v` and negatively sampled nodes `v'`,
            for each node `u` in the batch. Expected shape (1 + neg_size, dim, batch_size).
        indices_u : list of int
            List of node indices for each of the vectors in `vectors_u`.
        indices_v : list of lists of int
            Nested list of lists, each of which is a  list of node indices
            for each of the vectors in `vectors_v` for a specific node `u`.
        regularization_coeff : float, optional
            Coefficient to use for l2-regularization

        """
        self.vectors_u = vectors_u.T[np.newaxis, :, :]
        self.vectors_v = vectors_v
        self.indices_u = indices_u
        self.indices_v = indices_v
        self.regularization_coeff = regularization_coeff
        self.poincare_dists = None
        self.euclidean_dists = None
        self.norms_u = None
        self.norms_v = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.gradients_u = None
        self.distance_gradients_u = None
        self.gradients_v = None
        self.distance_gradients_v = None
        self.loss = None
        self._distances_computed = False
        self._gradients_computed = False
        self._distance_gradients_computed = False
        self._loss_computed = False

    def compute_all(self):
        """Convenience method to perform all computations."""
        self.compute_distances()
        self.compute_distance_gradients()
        self.compute_gradients()
        self.compute_loss()

    def compute_distances(self):
        """Compute and store norms, euclidean distances and poincare distances between input vectors."""
        if self._distances_computed:
            return
        euclidean_dists = np.linalg.norm(self.vectors_u - self.vectors_v, axis=1)
        norms_u = np.linalg.norm(self.vectors_u, axis=1)
        norms_v = np.linalg.norm(self.vectors_v, axis=1)
        alpha = 1 - norms_u ** 2
        beta = 1 - norms_v ** 2
        gamma = 1 + 2 * (euclidean_dists ** 2 / (alpha * beta))
        poincare_dists = np.arccosh(gamma)
        exp_negative_distances = np.exp(-poincare_dists)
        Z = exp_negative_distances.sum(axis=0)
        self.euclidean_dists = euclidean_dists
        self.poincare_dists = poincare_dists
        self.exp_negative_distances = exp_negative_distances
        self.Z = Z
        self.gamma = gamma
        self.norms_u = norms_u
        self.norms_v = norms_v
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._distances_computed = True

    def compute_gradients(self):
        """Compute and store gradients of loss function for all input vectors."""
        if self._gradients_computed:
            return
        self.compute_distances()
        self.compute_distance_gradients()
        gradients_v = -self.exp_negative_distances[:, np.newaxis, :] * self.distance_gradients_v
        gradients_v /= self.Z
        gradients_v[0] += self.distance_gradients_v[0]
        gradients_v[0] += self.regularization_coeff * 2 * self.vectors_v[0]
        gradients_u = -self.exp_negative_distances[:, np.newaxis, :] * self.distance_gradients_u
        gradients_u /= self.Z
        gradients_u = gradients_u.sum(axis=0)
        gradients_u += self.distance_gradients_u[0]
        assert not np.isnan(gradients_u).any()
        assert not np.isnan(gradients_v).any()
        self.gradients_u = gradients_u
        self.gradients_v = gradients_v
        self._gradients_computed = True

    def compute_distance_gradients(self):
        """Compute and store partial derivatives of poincare distance d(u, v) w.r.t all u and all v."""
        if self._distance_gradients_computed:
            return
        self.compute_distances()
        euclidean_dists_squared = self.euclidean_dists ** 2
        c_ = (4 / (self.alpha * self.beta * np.sqrt(self.gamma ** 2 - 1)))[:, np.newaxis, :]
        u_coeffs = ((euclidean_dists_squared + self.alpha) / self.alpha)[:, np.newaxis, :]
        distance_gradients_u = u_coeffs * self.vectors_u - self.vectors_v
        distance_gradients_u *= c_
        nan_gradients = self.gamma == 1
        if nan_gradients.any():
            distance_gradients_u.swapaxes(1, 2)[nan_gradients] = 0
        self.distance_gradients_u = distance_gradients_u
        v_coeffs = ((euclidean_dists_squared + self.beta) / self.beta)[:, np.newaxis, :]
        distance_gradients_v = v_coeffs * self.vectors_v - self.vectors_u
        distance_gradients_v *= c_
        if nan_gradients.any():
            distance_gradients_v.swapaxes(1, 2)[nan_gradients] = 0
        self.distance_gradients_v = distance_gradients_v
        self._distance_gradients_computed = True

    def compute_loss(self):
        """Compute and store loss value for the given batch of examples."""
        if self._loss_computed:
            return
        self.compute_distances()
        self.loss = -np.log(self.exp_negative_distances[0] / self.Z).sum()
        self._loss_computed = True