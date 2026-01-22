from parlai.core.agents import Agent
from parlai.utils.torch import padded_3d
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.misc import recursive_getattr
from parlai.utils.logging import logging
from .modules import (
import torch
def add_common_cmdline_args(argparser):
    """
    Add common command line args.
    """
    argparser.add_argument('-esz', '--embedding-size', type=int, default=300, help='Size of all embedding layers')
    argparser.add_argument('-nl', '--n-layers', type=int, default=2)
    argparser.add_argument('-hid', '--ffn-size', type=int, default=300, help='Hidden size of the FFN layers')
    argparser.add_argument('--dropout', type=float, default=0.0, help='Dropout used in Vaswani 2017.')
    argparser.add_argument('--attention-dropout', type=float, default=0.0, help='Dropout used after attention softmax.')
    argparser.add_argument('--relu-dropout', type=float, default=0.0, help='Dropout used after ReLU. From tensor2tensor.')
    argparser.add_argument('--n-heads', type=int, default=2, help='Number of multihead attention heads')
    argparser.add_argument('--learn-positional-embeddings', type='bool', default=False)
    argparser.add_argument('--embeddings-scale', type='bool', default=True)
    argparser.add_argument('--n-positions', type=int, default=None, hidden=True, help='Number of positional embeddings to learn. Defaults to truncate or 1024 if not provided.')
    argparser.add_argument('--n-segments', type=int, default=0, help='The number of segments that support the model. If zero no segment and no langs_embedding.')
    argparser.add_argument('--variant', choices={'aiayn', 'xlm', 'prelayernorm', 'bart'}, default='aiayn', help='Chooses locations of layer norms, etc. prelayernorm is used to match some fairseq models', recommended='xlm')
    argparser.add_argument('--activation', choices={'relu', 'gelu'}, default='relu', help='Nonlinear activation to use. AIAYN uses relu, but more recent papers prefer gelu.', recommended='gelu')
    argparser.add_argument('--output-scaling', type=float, default=1.0, help='scale the output of every transformer by this quantity.')
    argparser.add_argument('--share-word-embeddings', type='bool', default=True, help='Share word embeddings table for candidate and contextin the memory network')
    argparser.add_argument('-nel', '--n-encoder-layers', type=int, default=-1, help='This will overide the n-layers for asymmetrical transformers')
    argparser.add_argument('-ndl', '--n-decoder-layers', type=int, default=-1, help='This will overide the n-layers for asymmetrical transformers')
    argparser.add_argument('--model-parallel', type='bool', default=False, help='Shard the layers across multiple GPUs.')