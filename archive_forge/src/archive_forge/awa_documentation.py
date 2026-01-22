from typing import Any, Dict, List
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator
Compute query embeddings using AwaEmbedding.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        