import json
import urllib.request
from base64 import b64encode
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.utils import get_from_env, stringify_value
from langchain_community.document_loaders.base import BaseLoader


        Args:
            resource: The Modern Treasury resource to load.
            organization_id: The Modern Treasury organization ID. It can also be
               specified via the environment variable
               "MODERN_TREASURY_ORGANIZATION_ID".
            api_key: The Modern Treasury API key. It can also be specified via
               the environment variable "MODERN_TREASURY_API_KEY".
        