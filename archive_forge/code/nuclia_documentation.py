import json
import uuid
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.tools.nuclia.tool import NucliaUnderstandingAPI
Load documents.