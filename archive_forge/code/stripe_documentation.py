import json
import urllib.request
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.utils import get_from_env, stringify_dict
from langchain_community.document_loaders.base import BaseLoader
Initialize with a resource and an access token.

        Args:
            resource: The resource.
            access_token: The access token.
        