import collections
from typing import List, Optional, Union
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, add_end_docstrings, add_start_docstrings, logging
from ..bert.tokenization_bert import BertTokenizer
@add_end_docstrings(CUSTOM_DPR_READER_DOCSTRING)
class DPRReaderTokenizer(CustomDPRReaderTokenizerMixin, BertTokenizer):
    """
    Construct a DPRReader tokenizer.

    [`DPRReaderTokenizer`] is almost identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece. The difference is that is has three inputs strings: question, titles and texts that are
    combined to be fed to the [`DPRReader`] model.

    Refer to superclass [`BertTokenizer`] for usage examples and documentation concerning parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = READER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = READER_PRETRAINED_INIT_CONFIGURATION
    model_input_names = ['input_ids', 'attention_mask']