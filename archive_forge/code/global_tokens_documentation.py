from dataclasses import dataclass
from typing import Optional, Union
import torch
import torch.nn as nn
from xformers.components.attention import (
from xformers.components.attention.attention_patterns import (
from xformers.components.attention.core import scaled_dot_product_attention

        Global attention, as proposed for instance in BigBird_ or Longformer_.

        Global means in that case that the queries positively labelled in the ```attention_query_mask``` can attend
        to all the other queries. The queries negatively labelled in the ```attention_query_mask``` cannot attend to
        any other query.

        This implementation is sparse-aware, meaning that the empty attention parts will not be represented in memory.

        Args:
            dropout (float): probability of an element to be zeroed
            attention_query_mask (torch.Tensor): if true, this query can attend to all the others

        