from typing import List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
Load and return documents for each guide linked to from the device