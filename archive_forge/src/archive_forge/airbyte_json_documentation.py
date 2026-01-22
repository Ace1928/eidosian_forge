import json
from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document
from langchain_core.utils import stringify_dict
from langchain_community.document_loaders.base import BaseLoader
Path to the directory containing the json files.