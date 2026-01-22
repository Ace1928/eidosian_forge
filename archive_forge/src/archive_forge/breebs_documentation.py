from typing import List
import requests
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents.base import Document
from langchain_core.retrievers import BaseRetriever
Retrieve context for given query.
        Note that for time being there is no score.