from pathlib import Path
from typing import Any, Dict, List
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
class OpenVINOBgeEmbeddings(OpenVINOEmbeddings):
    """OpenVNO BGE embedding models.

    Bge Example:
        .. code-block:: python

            from langchain_community.embeddings import OpenVINOBgeEmbeddings

            model_name = "BAAI/bge-large-en"
            model_kwargs = {'device': 'CPU'}
            encode_kwargs = {'normalize_embeddings': True}
            ov = OpenVINOBgeEmbeddings(
                model_name_or_path=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """
    query_instruction: str = DEFAULT_QUERY_BGE_INSTRUCTION_EN
    'Instruction to use for embedding query.'
    embed_instruction: str = ''
    'Instruction to use for embedding document.'

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        if '-zh' in self.model_name_or_path:
            self.query_instruction = DEFAULT_QUERY_BGE_INSTRUCTION_ZH

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [self.embed_instruction + t.replace('\n', ' ') for t in texts]
        embeddings = self.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace('\n', ' ')
        embedding = self.encode(self.query_instruction + text, **self.encode_kwargs)
        return embedding.tolist()