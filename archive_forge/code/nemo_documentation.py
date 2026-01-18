from __future__ import annotations
import asyncio
import json
from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator
Call out to NeMo's embedding endpoint async for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        