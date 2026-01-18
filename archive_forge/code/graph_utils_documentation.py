from __future__ import annotations
import itertools
import operator
from typing import TYPE_CHECKING
import networkx as nx
import numpy as np
from monty.json import MSONable
Orders the SimpleGraphCycle.

        The ordering is performed such that the first node is the "lowest" one
        and the second node is the lowest one of the two neighbor nodes of the
        first node. If raise_on_fail is set to True a RuntimeError will be
        raised if the ordering fails.

        Args:
            raise_on_fail: If set to True, will raise a RuntimeError if the ordering fails.
        