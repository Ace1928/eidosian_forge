import math
from functools import partial
import torch
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn.utils.distributed import get_dim_for_local_rank

        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        