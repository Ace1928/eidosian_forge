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
def _get_tables_to_query(self, table_names: Optional[Union[List[str], str]]=None) -> Optional[List[str]]:
    """Get the tables names that need to be queried, after checking they exist."""
    if table_names is not None:
        if isinstance(table_names, list) and len(table_names) > 0 and (table_names[0] != ''):
            fixed_tables = [fix_table_name(table) for table in table_names]
            non_existing_tables = [table for table in fixed_tables if table not in self.table_names]
            if non_existing_tables:
                logger.warning('Table(s) %s not found in dataset.', ', '.join(non_existing_tables))
            tables = [table for table in fixed_tables if table not in non_existing_tables]
            return tables if tables else None
        if isinstance(table_names, str) and table_names != '':
            if table_names not in self.table_names:
                logger.warning('Table %s not found in dataset.', table_names)
                return None
            return [fix_table_name(table_names)]
    return self.table_names