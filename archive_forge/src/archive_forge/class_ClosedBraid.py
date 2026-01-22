from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise
class ClosedBraid(Link):
    """
    This is a convenience class for constructing closed braids.

    The constructor accepts either a single argument, which should be a list of
    integers to be passed to the Link constructor as the braid_closure
    parameter, or one or more integer arguments which will be packaged as a list
    and used as the braid_closure parameter.

    >>> B = ClosedBraid(1,-2,3)
    >>> B
    ClosedBraid(1, -2, 3)
    >>> B = ClosedBraid([1,-2,3]*3)
    >>> B
    ClosedBraid(1, -2, 3, 1, -2, 3, 1, -2, 3)
    """

    def __init__(self, *args, **kwargs):
        if args and 'braid_closure' not in kwargs:
            if len(args) == 1:
                self.braid_word = kwargs['braid_closure'] = tuple(args[0])
                args = ()
            elif isinstance(args[0], int):
                self.braid_word = kwargs['braid_closure'] = args
                args = ()
        Link.__init__(self, *args, **kwargs)

    def __repr__(self):
        return 'ClosedBraid%s' % str(self.braid_word)