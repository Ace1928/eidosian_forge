import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
def _filter_name(name):
    filtered_out_names = [MEMORY_EVENT_NAME, OUT_OF_MEMORY_EVENT_NAME, 'profiler::_record_function_enter', 'profiler::_record_function_enter_new', 'profiler::_record_function_exit', 'aten::is_leaf', 'aten::output_nr', 'aten::_version']
    return name in filtered_out_names