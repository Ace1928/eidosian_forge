from functools import lru_cache
import torch
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .modules import MemNN, opt_to_kwargs

        Build memory tensors.

        During building, will add time features to the memories if enabled.

        :param mems:
            list of length batchsize containing inner lists of 1D tensors
            containing the individual memories for each row in the batch.

        :returns:
            3d padded tensor of memories (bsz x num_mems x seqlen)
        