import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from ....modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
from ....modeling_utils import ModuleUtilsMixin
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings

        Returns:

        Examples:

        ```python
        # For example purposes. Not runnable.
        transformer = BertModel.from_pretrained("google-bert/bert-base-uncased")
        encoder = ImageEncoder(args)
        mmbt = MMBTModel(config, transformer, encoder)
        ```