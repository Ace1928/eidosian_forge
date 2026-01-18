import os
import pickle
import time
from typing import Iterable, List, Optional, Tuple
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends, strtobool
from .configuration_rag import RagConfig
from .tokenization_rag import RagTokenizer

        Retrieves documents for specified `question_hidden_states`.

        Args:
            question_input_ids (`List[List[int]]`) batch of input ids
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            prefix (`str`, *optional*):
                The prefix used by the generator's tokenizer.
            n_docs (`int`, *optional*):
                The number of docs retrieved per query.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.

        Returns: [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **context_input_ids** -- List of token ids to be fed to a model.

              [What are input IDs?](../glossary#input-ids)

            - **context_attention_mask** -- List of indices specifying which tokens should be attended to by the model
            (when `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

              [What are attention masks?](../glossary#attention-mask)

            - **retrieved_doc_embeds** -- List of embeddings of the retrieved documents
            - **doc_ids** -- List of ids of the retrieved documents
        