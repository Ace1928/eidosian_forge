import copy
from math import sqrt, log, exp
from itertools import cycle
import warnings
import numpy
from . import tools
class StrategyOnePlusLambda(object):
    """
    A CMA-ES strategy that uses the :math:`1 + \\lambda` paradigm ([Igel2007]_).

    :param parent: An iterable object that indicates where to start the
                   evolution. The parent requires a fitness attribute.
    :param sigma: The initial standard deviation of the distribution.
    :param lambda_: Number of offspring to produce from the parent.
                    (optional, defaults to 1)
    :param parameter: One or more parameter to pass to the strategy as
                      described in the following table. (optional)

    Other parameters can be provided as described in the next table

    +----------------+---------------------------+----------------------------+
    | Parameter      | Default                   | Details                    |
    +================+===========================+============================+
    | ``d``          | ``1.0 + N / (2.0 *        | Damping for step-size.     |
    |                | lambda_)``                |                            |
    +----------------+---------------------------+----------------------------+
    | ``ptarg``      | ``1.0 / (5 + sqrt(lambda_)| Target success rate.        |
    |                | / 2.0)``                  |                            |
    +----------------+---------------------------+----------------------------+
    | ``cp``         | ``ptarg * lambda_ / (2.0 +| Step size learning rate.   |
    |                | ptarg * lambda_)``        |                            |
    +----------------+---------------------------+----------------------------+
    | ``cc``         | ``2.0 / (N + 2.0)``       | Cumulation time horizon.   |
    +----------------+---------------------------+----------------------------+
    | ``ccov``       | ``2.0 / (N**2 + 6.0)``    | Covariance matrix learning |
    |                |                           | rate.                      |
    +----------------+---------------------------+----------------------------+
    | ``pthresh``    | ``0.44``                  | Threshold success rate.    |
    +----------------+---------------------------+----------------------------+

    .. [Igel2007] Igel, Hansen, Roth, 2007. Covariance matrix adaptation for
    multi-objective optimization. *Evolutionary Computation* Spring;15(1):1-28

    """

    def __init__(self, parent, sigma, **kargs):
        self.parent = parent
        self.sigma = sigma
        self.dim = len(self.parent)
        self.C = numpy.identity(self.dim)
        self.A = numpy.identity(self.dim)
        self.pc = numpy.zeros(self.dim)
        self.computeParams(kargs)
        self.psucc = self.ptarg

    def computeParams(self, params):
        """Computes the parameters depending on :math:`\\lambda`. It needs to
        be called again if :math:`\\lambda` changes during evolution.

        :param params: A dictionary of the manually set parameters.
        """
        self.lambda_ = params.get('lambda_', 1)
        self.d = params.get('d', 1.0 + self.dim / (2.0 * self.lambda_))
        self.ptarg = params.get('ptarg', 1.0 / (5 + sqrt(self.lambda_) / 2.0))
        self.cp = params.get('cp', self.ptarg * self.lambda_ / (2 + self.ptarg * self.lambda_))
        self.cc = params.get('cc', 2.0 / (self.dim + 2.0))
        self.ccov = params.get('ccov', 2.0 / (self.dim ** 2 + 6.0))
        self.pthresh = params.get('pthresh', 0.44)

    def generate(self, ind_init):
        """Generate a population of :math:`\\lambda` individuals of type
        *ind_init* from the current strategy.

        :param ind_init: A function object that is able to initialize an
                         individual from a list.
        :returns: A list of individuals.
        """
        arz = numpy.random.standard_normal((self.lambda_, self.dim))
        arz = self.parent + self.sigma * numpy.dot(arz, self.A.T)
        return [ind_init(a) for a in arz]

    def update(self, population):
        """Update the current covariance matrix strategy from the
        *population*.

        :param population: A list of individuals from which to update the
                           parameters.
        """
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        lambda_succ = sum((self.parent.fitness <= ind.fitness for ind in population))
        p_succ = float(lambda_succ) / self.lambda_
        self.psucc = (1 - self.cp) * self.psucc + self.cp * p_succ
        if self.parent.fitness <= population[0].fitness:
            x_step = (population[0] - numpy.array(self.parent)) / self.sigma
            self.parent = copy.deepcopy(population[0])
            if self.psucc < self.pthresh:
                self.pc = (1 - self.cc) * self.pc + sqrt(self.cc * (2 - self.cc)) * x_step
                self.C = (1 - self.ccov) * self.C + self.ccov * numpy.outer(self.pc, self.pc)
            else:
                self.pc = (1 - self.cc) * self.pc
                self.C = (1 - self.ccov) * self.C + self.ccov * (numpy.outer(self.pc, self.pc) + self.cc * (2 - self.cc) * self.C)
        self.sigma = self.sigma * exp(1.0 / self.d * (self.psucc - self.ptarg) / (1.0 - self.ptarg))
        self.A = numpy.linalg.cholesky(self.C)