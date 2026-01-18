import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph import Graph
from typing import Union, Dict, Any, Set
 This class is created to make sure PackedParams
    (e.g. LinearPackedParams, Conv2dPackedParams) to appear in state_dict
    so that we can serialize and deserialize quantized graph module with
    torch.save(m.state_dict()) and m.load_state_dict(state_dict)
    