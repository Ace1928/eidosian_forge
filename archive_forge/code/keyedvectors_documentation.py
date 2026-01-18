import logging
import sys
import itertools
import warnings
from numbers import Integral
from typing import Iterable
from numpy import (
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
A single vocabulary item, used internally for collecting per-word frequency/sampling info,
        and for constructing binary trees (incl. both word leaves and inner nodes).

        Retained for now to ease the loading of older models.
        