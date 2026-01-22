from typing import Dict, Tuple
from torch.distributed.checkpoint.metadata import (
from ._traverse import (
Restore the original nested state_dict according to ``mapping`` and the flattened ``state_dict``.