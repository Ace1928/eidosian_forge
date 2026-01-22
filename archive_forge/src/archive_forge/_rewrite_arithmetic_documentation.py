from numbers import Real
import numpy as np
from pyquil.quil import Program
from pyquil.quilatom import MemoryReference, Expression, Sub, Div
from pyquil.quilbase import (
from rpcq.messages import ParameterSpec, ParameterAref, RewriteArithmeticResponse
from typing import Dict, Union, List, no_type_check
Get a suitable MemoryReference for a given expression.