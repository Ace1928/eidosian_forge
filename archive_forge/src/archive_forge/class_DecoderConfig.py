from ...configuration_utils import PretrainedConfig
from ...utils import logging
class DecoderConfig(PretrainedConfig):
    """
    Configuration class for FSMT's decoder specific things. note: this is a private helper class
    """
    model_type = 'fsmt_decoder'

    def __init__(self, vocab_size=0, bos_token_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id