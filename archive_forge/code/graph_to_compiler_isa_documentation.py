from typing import Union
import numpy as np
from typing import List, Optional, cast
from pyquil.external.rpcq import (
import networkx as nx

    Signals an error when creating a ``CompilerISA`` from an ``nx.Graph``.
    This may raise as a consequence of unsupported gates.
    