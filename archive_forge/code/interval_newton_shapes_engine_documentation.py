from ..matrix import matrix, vector, mat_solve
from .. import snap
from ..sage_helper import _within_sage, sage_method

        Try Newton interval iterations, expanding the shape intervals
        until we can certify they contain a true solution.
        If succeeded, return True and write certified shapes to
        certified_shapes.
        Set verbose = True for printing additional information.
        