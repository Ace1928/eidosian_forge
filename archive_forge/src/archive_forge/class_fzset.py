import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
class fzset(frozenset):

    def __repr__(self):
        return '{%s}' % ', '.join(map(repr, self))