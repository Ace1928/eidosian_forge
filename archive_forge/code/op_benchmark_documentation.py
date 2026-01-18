import numpy as np
import torch
from torch.utils.benchmark import Timer
from torch.utils.benchmark.op_fuzzers.binary import BinaryOpFuzzer
from torch.utils.benchmark.op_fuzzers.unary import UnaryOpFuzzer
Builtin dict comparison will not compare numpy arrays.
    e.g.
        x = {"a": np.ones((2, 1))}
        x == x  # Raises ValueError
    