from .bi_encoder_ranker import BiEncoderRankerAgent  # NOQA
from .cross_encoder_ranker import CrossEncoderRankerAgent  # NOQA
from .both_encoder_ranker import BothEncoderRankerAgent  # NOQA
from parlai.core.torch_agent import TorchAgent
class BertRankerAgent(TorchAgent):
    """
    Abstract parent class for all Bert Ranker agents.
    """

    def __init__(self, opt, shared=None):
        raise RuntimeError('You must specify which ranker to use. Choices: \n-m bert_ranker/bi_encoder_ranker \n-m bert_ranker/cross_encoder_ranker \n-m bert_ranker/both_encoder_ranker')