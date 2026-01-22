from functools import wraps
from itertools import repeat
class DeltaPenalty(object):
    """This decorator returns penalized fitness for invalid individuals and the
    original fitness value for valid individuals. The penalized fitness is made
    of a constant factor *delta* added with an (optional) *distance* penalty. The
    distance function, if provided, shall return a value growing as the
    individual moves away the valid zone.

    :param feasibility: A function returning the validity status of any
                        individual.
    :param delta: Constant or array of constants returned for an invalid individual.
    :param distance: A function returning the distance between the individual
                     and a given valid point. The distance function can also return a sequence
                     of length equal to the number of objectives to affect multi-objective
                     fitnesses differently (optional, defaults to 0).
    :returns: A decorator for evaluation function.

    This function relies on the fitness weights to add correctly the distance.
    The fitness value of the ith objective is defined as

    .. math::

       f^\\mathrm{penalty}_i(\\mathbf{x}) = \\Delta_i - w_i d_i(\\mathbf{x})

    where :math:`\\mathbf{x}` is the individual, :math:`\\Delta_i` is a user defined
    constant and :math:`w_i` is the weight of the ith objective. :math:`\\Delta`
    should be worst than the fitness of any possible individual, this means
    higher than any fitness for minimization and lower than any fitness for
    maximization.

    See the :doc:`/tutorials/advanced/constraints` for an example.
    """

    def __init__(self, feasibility, delta, distance=None):
        self.fbty_fct = feasibility
        if not isinstance(delta, Sequence):
            self.delta = repeat(delta)
        else:
            self.delta = delta
        self.dist_fct = distance

    def __call__(self, func):

        @wraps(func)
        def wrapper(individual, *args, **kwargs):
            if self.fbty_fct(individual):
                return func(individual, *args, **kwargs)
            weights = tuple((1 if w >= 0 else -1 for w in individual.fitness.weights))
            dists = tuple((0 for w in individual.fitness.weights))
            if self.dist_fct is not None:
                dists = self.dist_fct(individual)
                if not isinstance(dists, Sequence):
                    dists = repeat(dists)
            return tuple((d - w * dist for d, w, dist in zip(self.delta, weights, dists)))
        return wrapper