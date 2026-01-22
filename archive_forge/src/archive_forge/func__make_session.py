import contextlib
import decimal
import uuid
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence, Union
from sqlalchemy import (
from sqlalchemy.ext.asyncio import (
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Query, Session, sessionmaker
from langchain.indexes.base import RecordManager
@contextlib.contextmanager
def _make_session(self) -> Generator[Session, None, None]:
    """Create a session and close it after use."""
    if isinstance(self.session_factory, async_sessionmaker):
        raise AssertionError('This method is not supported for async engines.')
    session = self.session_factory()
    try:
        yield session
    finally:
        session.close()