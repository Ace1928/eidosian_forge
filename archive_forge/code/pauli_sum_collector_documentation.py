import collections
from typing import cast, Dict, Optional, Union, TYPE_CHECKING
import numpy as np
from cirq import ops
from cirq.work import collector
Sums up the sampled expectations, weighted by their coefficients.