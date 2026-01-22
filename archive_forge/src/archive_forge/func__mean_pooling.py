from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra
@staticmethod
def _mean_pooling(outputs: Any, attention_mask: Any) -> Any:
    try:
        import torch
    except ImportError as e:
        raise ImportError('Unable to import torch, please install with `pip install -U torch`.') from e
    if isinstance(outputs, dict):
        token_embeddings = outputs['last_hidden_state']
    else:
        token_embeddings = outputs[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-09)
    return sum_embeddings / sum_mask