import json
import re
import zipfile
from abc import ABC
from pathlib import Path
from typing import Iterator, List, Set, Tuple
from langchain_community.docstore.document import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
Get the relationships of a page and the relationships of its relationships,
        etc... recursively.
        Pages are based on other pages (ex: background page),
        so we need to get all the relationships to get all the content of a single page.
        