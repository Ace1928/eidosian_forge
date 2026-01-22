import itertools
from warnings import warn
import torch
import torch.cuda
from torch.autograd import (
from torch.autograd.profiler_util import (
def _get_record_key(record):
    """Return a tuple for correlating start and end records in `_parse_legacy_records`."""
    return (record.handle(), record.node_id())