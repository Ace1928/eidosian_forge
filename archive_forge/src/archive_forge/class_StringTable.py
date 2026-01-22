import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
class StringTable(defaultdict):

    def __missing__(self, key):
        self[key] = torch._C._demangle(key) if len(key) > 1 else key
        return self[key]