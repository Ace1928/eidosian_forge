import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
def _format_memory(nbytes):
    """Return a formatted memory size string."""
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if abs(nbytes) >= GB:
        return f'{nbytes * 1.0 / GB:.2f} Gb'
    elif abs(nbytes) >= MB:
        return f'{nbytes * 1.0 / MB:.2f} Mb'
    elif abs(nbytes) >= KB:
        return f'{nbytes * 1.0 / KB:.2f} Kb'
    else:
        return str(nbytes) + ' b'