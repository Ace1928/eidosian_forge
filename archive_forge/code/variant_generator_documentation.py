import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
Assigns a value to a nested dictionary.

    Handles the special case of tuples, in which case the tuples
    will be re-constructed to accomodate the updated value.
    