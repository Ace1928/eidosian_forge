import os
import tempfile
from abc import ABC
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
Check if the url is valid.