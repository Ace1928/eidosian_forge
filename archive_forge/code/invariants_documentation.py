from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise

        A quick test:

            sage: L = Link('K13n100')
            sage: L._sage_()
            Knot represented by 13 crossings
        