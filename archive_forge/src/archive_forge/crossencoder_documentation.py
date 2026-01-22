from .modules import TransformerEncoder
from .modules import get_n_positions_from_options
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .transformer import TransformerRankerAgent
from parlai.utils.torch import concat_without_padding
import torch

        Scores each concatenation text + candidate.
        