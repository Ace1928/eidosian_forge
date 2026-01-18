import functools
import logging
from typing import cast, List, Tuple
import sympy
import torch
from torch._inductor.select_algorithm import realize_inputs
from torch._inductor.virtualized import V
from ..utils import ceildiv as cdiv, next_power_of_2

    Common arg processing for mm,bmm,addmm,etc
    