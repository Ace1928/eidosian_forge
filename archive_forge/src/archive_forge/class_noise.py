from math import hypot, sqrt
from functools import wraps
from itertools import repeat
class noise(object):
    """Decorator for evaluation functions, it evaluates the objective function
    and adds noise by calling the function(s) provided in the *noise*
    argument. The noise functions are called without any argument, consider
    using the :class:`~deap.base.Toolbox` or Python's
    :func:`functools.partial` to provide any required argument. If a single
    function is provided it is applied to all objectives of the evaluation
    function. If a list of noise functions is provided, it must be of length
    equal to the number of objectives. The noise argument also accept
    :obj:`None`, which will leave the objective without noise.

    This decorator adds a :func:`noise` method to the decorated
    function.
    """

    def __init__(self, noise):
        try:
            self.rand_funcs = tuple(noise)
        except TypeError:
            self.rand_funcs = repeat(noise)

    def __call__(self, func):

        @wraps(func)
        def wrapper(individual, *args, **kargs):
            result = func(individual, *args, **kargs)
            noisy = list()
            for r, f in zip(result, self.rand_funcs):
                if f is None:
                    noisy.append(r)
                else:
                    noisy.append(r + f())
            return tuple(noisy)
        wrapper.noise = self.noise
        return wrapper

    def noise(self, noise):
        """Set the current noise to *noise*. After decorating the
        evaluation function, this function will be available directly from
        the function object. ::

            prand = functools.partial(random.gauss, mu=0.0, sigma=1.0)

            @noise(prand)
            def evaluate(individual):
                return sum(individual),

            # This will remove noise from the evaluation function
            evaluate.noise(None)
        """
        try:
            self.rand_funcs = tuple(noise)
        except TypeError:
            self.rand_funcs = repeat(noise)