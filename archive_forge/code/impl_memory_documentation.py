import contextlib
import copy
import itertools
import posixpath as pp
import fasteners
from taskflow import exceptions as exc
from taskflow.persistence import path_based
from taskflow.types import tree
This just wraps a global write-lock.