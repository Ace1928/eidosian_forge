from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.device_mesh import DeviceMesh

        gen_fake_kwargs: generate fake kwargs for the operator, this is mainly used
            by sharding propagation rules to generate fake kwargs for the operator
            to run the local tensor operator and get the output spec.
        