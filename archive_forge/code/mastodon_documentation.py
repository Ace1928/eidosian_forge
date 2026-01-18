from __future__ import annotations
import os
from typing import (
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Format toots into documents.

        Adding user info, and selected toot fields into the metadata.
        