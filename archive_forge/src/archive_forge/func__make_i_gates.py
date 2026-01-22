from typing import Union
import numpy as np
from typing import List, Optional, cast
from pyquil.external.rpcq import (
import networkx as nx
def _make_i_gates() -> List[GateInfo]:
    return [GateInfo(operator=Supported1QGate.I, parameters=[], arguments=['_'])]