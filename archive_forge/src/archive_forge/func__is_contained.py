from dataclasses import dataclass, field
from collections import defaultdict
import copy
import torch
from torch.fx import (
from torch.fx._compatibility import compatibility
from typing import Dict, List, Set, Any, Union, Tuple
import logging
import os
def _is_contained(self, nodes_map: Dict[Node, Node]) -> bool:
    lookup: Dict[Node, Node] = {gn: pn for pn, gn in nodes_map.items() if pn.op != 'placeholder'}
    for gn, pn in lookup.items():
        if pn in self.pattern_returning_nodes:
            continue
        for user in gn.users:
            if user not in lookup:
                return False
    return True