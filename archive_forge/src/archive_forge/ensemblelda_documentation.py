import logging
import os
from multiprocessing import Process, Pipe, ProcessError
import importlib
from typing import Set, Optional, List
import numpy as np
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from gensim import utils
from gensim.models import ldamodel, ldamulticore, basemodel
from gensim.utils import SaveLoad
Extend the cluster in one direction.

            Results are accumulated to ``self.results``.

            Parameters
            ----------
            topic_index : int
                The topic that might be added to the existing cluster, or which might create a new cluster if necessary.
            current_label : int
                The label of the cluster that might be suitable for ``topic_index``

            