import os
from typing import Any, Dict, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.base import BaseLoader
Create a RSpace client.