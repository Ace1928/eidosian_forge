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
@staticmethod
def _build_index(config):
    if config.index_name == 'legacy':
        return LegacyIndex(config.retrieval_vector_size, config.index_path or LEGACY_INDEX_PATH)
    elif config.index_name == 'custom':
        return CustomHFIndex.load_from_disk(vector_size=config.retrieval_vector_size, dataset_path=config.passages_path, index_path=config.index_path)
    else:
        return CanonicalHFIndex(vector_size=config.retrieval_vector_size, dataset_name=config.dataset, dataset_split=config.dataset_split, index_name=config.index_name, index_path=config.index_path, use_dummy_dataset=config.use_dummy_dataset)