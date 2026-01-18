from __future__ import with_statement
from contextlib import contextmanager
import collections.abc
import logging
import warnings
import numbers
from html.entities import name2codepoint as n2cp
import pickle as _pickle
import re
import unicodedata
import os
import random
import itertools
import tempfile
from functools import wraps
import multiprocessing
import shutil
import sys
import subprocess
import inspect
import heapq
from copy import deepcopy
from datetime import datetime
import platform
import types
import numpy as np
import scipy.sparse
from smart_open import open
from gensim import __version__ as gensim_version
@deprecated('Function will be removed in 4.0.0')
def toptexts(query, texts, index, n=10):
    """Debug fnc to help inspect the top `n` most similar documents (according to a similarity index `index`),
    to see if they are actually related to the query.

    Parameters
    ----------
    query : {list of (int, number), numpy.ndarray}
        vector OR BoW (list of tuples)
    texts : str
        object that can return something insightful for each document via `texts[docid]`,
        such as its fulltext or snippet.
    index : any
        A instance from from :mod:`gensim.similarity.docsim`.

    Return
    ------
    list
        a list of 3-tuples (docid, doc's similarity to the query, texts[docid])

    """
    sims = index[query]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return [(topid, topcosine, texts[topid]) for topid, topcosine in sims[:n]]