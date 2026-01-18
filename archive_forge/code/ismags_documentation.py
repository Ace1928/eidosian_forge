import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps

        Processes ordered pair partitions as per the reference paper. Finds and
        returns all permutations and cosets that leave the graph unchanged.
        