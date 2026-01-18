import email
import logging
from pathlib import Path
from typing import Dict, Iterator, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Load MHTML document into document objects.