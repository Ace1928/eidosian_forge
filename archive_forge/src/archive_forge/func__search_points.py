from typing import Any, Iterable, List, Optional, Tuple
from uuid import uuid4
import numpy as np
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
def _search_points(self, embedding: List[float], k: int=4) -> List[dict]:
    """Search points."""
    if self.distance_strategy == DistanceStrategy.COSINE:
        vec = np.array(embedding)
        vec = vec / np.linalg.norm(vec)
        embedding = vec.tolist()
    payload = {'vector': embedding, 'limit': k}
    response = requests.post(SemaDB.BASE_URL + f'/collections/{self.collection_name}/points/search', json=payload, headers=self.headers)
    if response.status_code != 200:
        raise ValueError(f'Error searching: {response.text}')
    return response.json()['points']