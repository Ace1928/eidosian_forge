import copy
from collections import OrderedDict
from contextlib import contextmanager
from threading import RLock
from typing import Optional
Returns a list of objects in the annotated queue