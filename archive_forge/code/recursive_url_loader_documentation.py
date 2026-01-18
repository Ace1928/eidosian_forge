from __future__ import annotations
import asyncio
import logging
import re
from typing import (
import requests
from langchain_core.documents import Document
from langchain_core.utils.html import extract_sub_links
from langchain_community.document_loaders.base import BaseLoader
Lazy load web pages.
        When use_async is True, this function will not be lazy,
        but it will still work in the expected way, just not lazy.