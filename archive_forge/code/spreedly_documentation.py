import json
import urllib.request
from typing import List
from langchain_core.documents import Document
from langchain_core.utils import stringify_dict
from langchain_community.document_loaders.base import BaseLoader
Initialize with an access token and a resource.

        Args:
            access_token: The access token.
            resource: The resource.
        