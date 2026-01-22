from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
class AlephAlphaSymmetricSemanticEmbedding(AlephAlphaAsymmetricSemanticEmbedding):
    """Symmetric version of the Aleph Alpha's semantic embeddings.

    The main difference is that here, both the documents and
    queries are embedded with a SemanticRepresentation.Symmetric
    Example:
        .. code-block:: python

            from aleph_alpha import AlephAlphaSymmetricSemanticEmbedding

            embeddings = AlephAlphaAsymmetricSemanticEmbedding(
                normalize=True, compress_to_size=128
            )
            text = "This is a test text"

            doc_result = embeddings.embed_documents([text])
            query_result = embeddings.embed_query(text)
    """

    def _embed(self, text: str) -> List[float]:
        try:
            from aleph_alpha_client import Prompt, SemanticEmbeddingRequest, SemanticRepresentation
        except ImportError:
            raise ValueError('Could not import aleph_alpha_client python package. Please install it with `pip install aleph_alpha_client`.')
        query_params = {'prompt': Prompt.from_text(text), 'representation': SemanticRepresentation.Symmetric, 'compress_to_size': self.compress_to_size, 'normalize': self.normalize, 'contextual_control_threshold': self.contextual_control_threshold, 'control_log_additive': self.control_log_additive}
        query_request = SemanticEmbeddingRequest(**query_params)
        query_response = self.client.semantic_embed(request=query_request, model=self.model)
        return query_response.embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Aleph Alpha's Document endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        document_embeddings = []
        for text in texts:
            document_embeddings.append(self._embed(text))
        return document_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to Aleph Alpha's asymmetric, query embedding endpoint
        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._embed(text)