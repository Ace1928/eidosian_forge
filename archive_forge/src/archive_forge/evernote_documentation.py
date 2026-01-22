import hashlib
import logging
from base64 import b64decode
from pathlib import Path
from time import strptime
from typing import Any, Dict, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Parse Evernote xml.