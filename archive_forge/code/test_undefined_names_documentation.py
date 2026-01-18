import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip

        A Name node with an unrecognized context results in a RuntimeError being
        raised.
        