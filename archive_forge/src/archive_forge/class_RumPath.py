import logging
import math
import numpy as np
from ase.utils import longsum
class RumPath:
    """Describes a curved search path, taking into account information
    about (near-) rigid unit motions (RUMs).

    One can tag sub-molecules of the system, which are collections of
    particles that form a (near-)rigid unit. Let x1, ... xn be the positions
    of one such molecule, then we construct a path of the form
    xi(t) = xi(0) + (exp(K t) - I) yi + t wi + t c
    where yi = xi - <x>, c = <g> is a rigid translation, K is anti-symmetric
    so that exp(tK) yi denotes a rotation about the centre of mass, and wi
    is the remainind stretch of the molecule.

    The following variables are stored:
     * rotation_factors : array of acceleration factors
     * rigid_units : array of molecule indices
     * stretch : w
     * K : list of K matrices
     * y : list of y-vectors
    """

    def __init__(self, x_start, dirn, rigid_units, rotation_factors):
        """Initialise a `RumPath`

        Args:
          x_start : vector containing the positions in d x nAt shape
          dirn : search direction, same shape as x_start vector
          rigid_units : array of arrays of molecule indices
          rotation_factors : factor by which the rotation of each molecular
                             is accelerated; array of scalars, same length as
                             rigid_units
        """
        if not have_scipy:
            raise RuntimeError('RumPath depends on scipy, which could not be imported')
        self.rotation_factors = rotation_factors
        self.rigid_units = rigid_units
        self.K = []
        self.y = []
        w = dirn.copy().reshape([3, len(dirn) / 3])
        X = x_start.reshape([3, len(dirn) / 3])
        for I in rigid_units:
            x = X[:, I]
            y = x - x.mean(0).T
            g = w[:, I]
            f = g - g.mean(0).T
            A = np.zeros((3, 3))
            b = np.zeros(3)
            for j in range(len(I)):
                Yj = np.array([[y[1, j], 0.0, -y[2, j]], [-y[0, j], y[2, j], 0.0], [0.0, -y[1, j], y[0, j]]])
                A += np.dot(Yj.T, Yj)
                b += np.dot(Yj.T, f[:, j])
            N = nullspace(A)
            b -= np.dot(np.dot(N, N.T), b)
            A += np.dot(N, N.T)
            k = scipy.linalg.solve(A, b, sym_pos=True)
            K = np.array([[0.0, k[0], -k[2]], [-k[0], 0.0, k[1]], [k[2], -k[1], 0.0]])
            w[:, I] -= np.dot(K, y)
            self.K.append(K)
            self.y.append(y)
        self.stretch = w

    def step(self, alpha):
        """perform a step in the line-search, given a step-length alpha

        Args:
          alpha : step-length

        Returns:
          s : update for positions
        """
        s = alpha * self.stretch
        for I, K, y, rf in zip(self.rigid_units, self.K, self.y, self.rotation_factors):
            aK = alpha * rf * K
            s[:, I] += np.dot(aK, y + 0.5 * np.dot(aK, y + 1 / 3.0 * np.dot(aK, y)))
        return s.ravel()