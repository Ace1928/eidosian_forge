import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def torch_to_numpy(x):
    return x.detach().cpu().numpy()