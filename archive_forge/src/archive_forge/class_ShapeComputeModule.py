from typing import List, Optional
import torch
from torch.backends._nnapi.serializer import _NnapiSerializer
class ShapeComputeModule(torch.nn.Module):
    """Code-gen-ed module for tensor shape computation.

        module.prepare will mutate ser_model according to the computed operand
        shapes, based on the shapes of args.  Returns a list of output templates.
        """
    pass