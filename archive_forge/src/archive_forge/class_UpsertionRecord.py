import contextlib
import decimal
import uuid
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence, Union
from sqlalchemy import (
from sqlalchemy.ext.asyncio import (
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Query, Session, sessionmaker
from langchain.indexes.base import RecordManager
class UpsertionRecord(Base):
    """Table used to keep track of when a key was last updated."""
    __tablename__ = 'upsertion_record'
    uuid = Column(String, index=True, default=lambda: str(uuid.uuid4()), primary_key=True, nullable=False)
    key = Column(String, index=True)
    namespace = Column(String, index=True, nullable=False)
    group_id = Column(String, index=True, nullable=True)
    updated_at = Column(Float, index=True)
    __table_args__ = (UniqueConstraint('key', 'namespace', name='uix_key_namespace'), Index('ix_key_namespace', 'key', 'namespace'))