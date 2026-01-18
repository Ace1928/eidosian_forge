import datetime
import os
from typing import Any, List, Optional, Union
import torch
import torch.distributed as dist
from torch import Tensor
from typing_extensions import Self, override
from lightning_fabric.plugins.collectives.collective import Collective
from lightning_fabric.utilities.types import CollectibleGroup, RedOpType, ReduceOp
Collective operations using `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`__.

    .. warning:: This is an :ref:`experimental <versioning:Experimental API>` feature which is still in development.

    