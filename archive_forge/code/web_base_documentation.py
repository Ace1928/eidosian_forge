import asyncio
import logging
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
import aiohttp
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Load text from the urls in web_path async into Documents.