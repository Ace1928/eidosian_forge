from __future__ import annotations
import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union
import aiohttp
import requests
from aiohttp import ServerTimeoutError
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator, validator
from requests.exceptions import Timeout
def _get_schema_for_tables(self, table_names: List[str]) -> str:
    """Create a string of the table schemas for the supplied tables."""
    schemas = [schema for table, schema in self.schemas.items() if table in table_names]
    return ', '.join(schemas)