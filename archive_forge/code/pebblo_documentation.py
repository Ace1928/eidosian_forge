from __future__ import annotations
import logging
import os
import pathlib
import platform
from typing import Optional, Tuple
from langchain_core.env import get_runtime_environment
from langchain_core.pydantic_v1 import BaseModel
from langchain_community.document_loaders.base import BaseLoader
Fetch local runtime ip address.

    Returns:
        str: IP address
    