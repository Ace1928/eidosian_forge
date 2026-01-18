import json
import os
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import regex as re
from ....file_utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available
from ....tokenization_utils import AddedToken, PreTrainedTokenizer
from ....tokenization_utils_base import ENCODE_KWARGS_DOCSTRING, BatchEncoding, TextInput, TruncationStrategy
from ....utils import logging
def truncate_table_rows(self, table_content: Dict, question: str, answer: Optional[Union[str, List[str]]]=None, max_length=None):
    """
        Args:
        table_content:
            {"header": xxx, "rows": xxx, "id" (Optionally): xxx}

        question:
            natural language sentence

        answer:
            if for training, is the supervision; otherwise will be empty
        """
    delete_ratio, remain_token_len = self.estimate_delete_ratio(table_content, question, max_length)
    self.delete_unrelated_rows(table_content, question, answer, delete_ratio)
    maximum_keep_rows = 0
    for ind, row_example in enumerate(table_content['rows']):
        value_string = self.table_linearize.process_row(row_example, ind + 1)
        value_token_len = len(self.tokenize(value_string))
        if value_token_len > remain_token_len:
            break
        remain_token_len -= value_token_len
        maximum_keep_rows += 1
    del table_content['rows'][maximum_keep_rows:]