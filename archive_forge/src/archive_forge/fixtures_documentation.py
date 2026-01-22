import itertools
from contextlib import ExitStack
Kill a test with sigalarm if it runs too long.

    Only works on Unix at present.
    