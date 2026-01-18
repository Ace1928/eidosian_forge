import collections
import json
import os
import re
from typing import List, Optional, Tuple, Union
import numpy as np
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
from ...utils import PaddingStrategy, logging

        The tokenizer returns token_type_ids as separators between the Prefix part and the rest.
        token_type_ids is 1 for the Prefix part and 0 for the rest of the token.

        Example:
        ```python
        >>> from transformers import GPTSanJapaneseTokenizer

        >>> tokenizer = GPTSanJapaneseTokenizer.from_pretrained("Tanrei/GPTSAN-japanese")
        >>> x_token = tokenizer("ｱｲｳｴ")
        >>> # input_ids:      | SOT | SEG | ｱ | ｲ | ｳ | ｴ |
        >>> # token_type_ids: | 1   | 0   | 0 | 0 | 0 | 0 |

        >>> x_token = tokenizer("", prefix_text="ｱｲｳｴ")
        >>> # input_ids:      | SOT | ｱ | ｲ | ｳ | ｴ | SEG |
        >>> # token_type_ids: | 1   | 1 | 1 | 1 | 1 | 0  |

        >>> x_token = tokenizer("ｳｴ", prefix_text="ｱｲ")
        >>> # input_ids:      | SOT | ｱ | ｲ | SEG | ｳ | ｴ |
        >>> # token_type_ids: | 1   | 1 | 1 | 0   | 0 | 0 |
        ```