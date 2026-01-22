import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (

        Helper function used by read_tuple_value and read_set_value.
        