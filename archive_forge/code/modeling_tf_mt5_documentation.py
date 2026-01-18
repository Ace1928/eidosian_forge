from ...utils import logging
from ..t5.modeling_tf_t5 import TFT5EncoderModel, TFT5ForConditionalGeneration, TFT5Model
from .configuration_mt5 import MT5Config

    This class overrides [`TFT5EncoderModel`]. Please check the superclass for the appropriate documentation alongside
    usage examples.

    Examples:

    ```python
    >>> from transformers import TFMT5EncoderModel, AutoTokenizer

    >>> model = TFMT5EncoderModel.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> input_ids = tokenizer(article, return_tensors="tf").input_ids
    >>> outputs = model(input_ids)
    >>> hidden_state = outputs.last_hidden_state
    ```