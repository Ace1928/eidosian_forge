from __future__ import annotations
import contextlib
import enum
import logging
import uuid
from typing import (
import numpy as np
import sqlalchemy
from sqlalchemy import delete, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session
from sqlalchemy.sql import quoted_name
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
@classmethod
def __get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
    connection_string: str = get_from_dict_or_env(data=kwargs, key='connection_string', env_key='LANTERN_CONNECTION_STRING')
    if not connection_string:
        raise ValueError('Postgres connection string is requiredEither pass it as `connection_string` parameteror set the LANTERN_CONNECTION_STRING variable.')
    return connection_string