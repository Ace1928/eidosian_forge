from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import sys
import time
import weakref
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib import (
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle
def _iter_collection_raw_paths(self, master_transform, paths, all_transforms):
    """
        Helper method (along with `_iter_collection`) to implement
        `draw_path_collection` in a memory-efficient manner.

        This method yields all of the base path/transform combinations, given a
        master transform, a list of paths and list of transforms.

        The arguments should be exactly what is passed in to
        `draw_path_collection`.

        The backend should take each yielded path and transform and create an
        object that can be referenced (reused) later.
        """
    Npaths = len(paths)
    Ntransforms = len(all_transforms)
    N = max(Npaths, Ntransforms)
    if Npaths == 0:
        return
    transform = transforms.IdentityTransform()
    for i in range(N):
        path = paths[i % Npaths]
        if Ntransforms:
            transform = Affine2D(all_transforms[i % Ntransforms])
        yield (path, transform + master_transform)