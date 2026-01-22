import copy
from math import sqrt, log, exp
from itertools import cycle
import warnings
import numpy
from . import tools
def _infeasible_update(self, individual):
    if not hasattr(individual.fitness, 'constraint_violation'):
        return
    if self.constraint_vecs is None:
        shape = (len(individual.fitness.constraint_violation), self.dim)
        self.constraint_vecs = numpy.zeros(shape)
    for i in range(self.constraint_vecs.shape[0]):
        if individual.fitness.constraint_violation[i]:
            self.constraint_vecs[i] = (1 - self.cconst) * self.constraint_vecs[i] + self.cconst * individual._y
    W = numpy.dot(self.invA, self.constraint_vecs.T).T
    constraint_violation = numpy.sum(individual.fitness.constraint_violation)
    A_prime = self.A - self.beta / constraint_violation * numpy.sum(list((numpy.outer(self.constraint_vecs[i], W[i]) / numpy.dot(W[i], W[i]) for i in range(self.constraint_vecs.shape[0]) if individual.fitness.constraint_violation[i])), axis=0)
    try:
        self.invA = numpy.linalg.inv(A_prime)
    except numpy.linalg.LinAlgError:
        warnings.warn('Singular matrix inversion, invalid update in CMA-ES ignored', RuntimeWarning)
    else:
        self.A = A_prime