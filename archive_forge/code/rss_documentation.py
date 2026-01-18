import logging
from typing import Any, Iterator, List, Optional, Sequence
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.news import NewsURLLoader
Initialize with urls or OPML.