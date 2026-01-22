from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra
@staticmethod
def _cls_pooling(outputs: Any) -> Any:
    if isinstance(outputs, dict):
        token_embeddings = outputs['last_hidden_state']
    else:
        token_embeddings = outputs[0]
    return token_embeddings[:, 0]