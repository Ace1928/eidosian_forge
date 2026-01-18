from snappy import snap
from snappy.sage_helper import _within_sage, sage_method

        Try Krawczyk iterations (i.e., expanding the shape intervals [z]
        by the Krawczyk interval K(z0, [z], f)) until we can certify they
        contain a true solution.

        If succeeded, return True and write certified shapes to
        certified_shapes.
        Set verbose = True for printing additional information.
        