import itertools
import math
import operator
import random
from functools import reduce
library, props
        Given a Library and the list of Propery evaluators,
        optimize the library.
        The library is modified in place by removing building blocks
        (sidechains) that are not likely to pass the property
        criteria.
        