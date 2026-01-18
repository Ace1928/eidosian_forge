from typing import Dict, Optional, Set
import torch
from torch._ops import OpOverload, OpOverloadPacket, HigherOrderOperator
from torch._export.error import InternalError
from torch._export.pass_base import _ExportPassBase

    Our backend expects pure functional operators. For efficiency
    purposes, we keep view ops around while functionalizing the exported
    program. This pass replaces view ops with view copy ops for backends that
    need AOT memory planning.
    