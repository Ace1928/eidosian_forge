from .bert_dictionary import BertDictionaryAgent
from .bi_encoder_ranker import BiEncoderRankerAgent
from .cross_encoder_ranker import CrossEncoderRankerAgent
from .helpers import add_common_args
from parlai.core.torch_agent import TorchAgent, Output, Batch

        We pass the batch first in the biencoder, then filter with crossencoder.
        