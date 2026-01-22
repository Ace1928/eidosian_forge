from functools import wraps
from itertools import repeat
class ClosestValidPenalty(object):
    """This decorator returns penalized fitness for invalid individuals and the
    original fitness value for valid individuals. The penalized fitness is made
    of the fitness of the closest valid individual added with a weighted
    (optional) *distance* penalty. The distance function, if provided, shall
    return a value growing as the individual moves away the valid zone.

    :param feasibility: A function returning the validity status of any
                        individual.
    :param feasible: A function returning the closest feasible individual
                     from the current invalid individual.
    :param alpha: Multiplication factor on the distance between the valid and
                  invalid individual.
    :param distance: A function returning the distance between the individual
                     and a given valid point. The distance function can also return a sequence
                     of length equal to the number of objectives to affect multi-objective
                     fitnesses differently (optional, defaults to 0).
    :returns: A decorator for evaluation function.

    This function relies on the fitness weights to add correctly the distance.
    The fitness value of the ith objective is defined as

    .. math::

       f^\\mathrm{penalty}_i(\\mathbf{x}) = f_i(\\operatorname{valid}(\\mathbf{x})) - \\\\alpha w_i d_i(\\operatorname{valid}(\\mathbf{x}), \\mathbf{x})

    where :math:`\\mathbf{x}` is the individual,
    :math:`\\operatorname{valid}(\\mathbf{x})` is a function returning the closest
    valid individual to :math:`\\mathbf{x}`, :math:`\\\\alpha` is the distance
    multiplicative factor and :math:`w_i` is the weight of the ith objective.
    """

    def __init__(self, feasibility, feasible, alpha, distance=None):
        self.fbty_fct = feasibility
        self.fbl_fct = feasible
        self.alpha = alpha
        self.dist_fct = distance

    def __call__(self, func):

        @wraps(func)
        def wrapper(individual, *args, **kwargs):
            if self.fbty_fct(individual):
                return func(individual, *args, **kwargs)
            f_ind = self.fbl_fct(individual)
            f_fbl = func(f_ind, *args, **kwargs)
            weights = tuple((1.0 if w >= 0 else -1.0 for w in individual.fitness.weights))
            if len(weights) != len(f_fbl):
                raise IndexError('Fitness weights and computed fitness are of different size.')
            dists = tuple((0 for w in individual.fitness.weights))
            if self.dist_fct is not None:
                dists = self.dist_fct(f_ind, individual)
                if not isinstance(dists, Sequence):
                    dists = repeat(dists)
            return tuple((f - w * self.alpha * d for f, w, d in zip(f_fbl, weights, dists)))
        return wrapper