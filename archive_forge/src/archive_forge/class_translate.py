from math import hypot, sqrt
from functools import wraps
from itertools import repeat
class translate(object):
    """Decorator for evaluation functions, it translates the objective
    function by *vector* which should be the same length as the individual
    size. When called the decorated function should take as first argument the
    individual to be evaluated. The inverse translation vector is actually
    applied to the individual and the resulting list is given to the
    evaluation function. Thus, the evaluation function shall not be expecting
    an individual as it will receive a plain list.

    This decorator adds a :func:`translate` method to the decorated function.
    """

    def __init__(self, vector):
        self.vector = vector

    def __call__(self, func):

        @wraps(func)
        def wrapper(individual, *args, **kargs):
            return func([v - t for v, t in zip(individual, self.vector)], *args, **kargs)
        wrapper.translate = self.translate
        return wrapper

    def translate(self, vector):
        """Set the current translation to *vector*. After decorating the
        evaluation function, this function will be available directly from
        the function object. ::

            @translate([0.25, 0.5, ..., 0.1])
            def evaluate(individual):
                return sum(individual),

            # This will cancel the translation
            evaluate.translate([0.0, 0.0, ..., 0.0])
        """
        self.vector = vector