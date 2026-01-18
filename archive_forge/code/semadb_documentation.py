from typing import Any, Iterable, List, Optional, Tuple
from uuid import uuid4
import numpy as np
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
Return VectorStore initialized from texts and embeddings.