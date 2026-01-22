import copy
import logging
import re
from collections.abc import Mapping
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy
import random
from ray.tune.search.sample import Categorical, Domain, Function, RandomState
from ray.util.annotations import DeveloperAPI, PublicAPI
@DeveloperAPI
class RecursiveDependencyError(Exception):

    def __init__(self, msg: str):
        Exception.__init__(self, msg)