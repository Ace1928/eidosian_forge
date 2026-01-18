from pygsp import utils
from . import Filter  # prevent circular import in Python < 3.5
Design a Gabor filter bank.

    Design a filter bank where the kernel *k* is placed at each graph
    frequency.

    Parameters
    ----------
    G : graph
    k : lambda function
        kernel

    Examples
    --------
    >>> G = graphs.Logo()
    >>> k = lambda x: x / (1. - x)
    >>> g = filters.Gabor(G, k);

    