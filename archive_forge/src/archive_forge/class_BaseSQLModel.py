import datetime
import contextlib
from sqlalchemy import func
from sqlalchemy import delete as sqlalchemy_delete
from sqlalchemy import update as sqlalchemy_update
from sqlalchemy import exists as sqlalchemy_exists
from sqlalchemy import insert as sqlalchemy_insert
from sqlalchemy.future import select
from sqlalchemy.sql.expression import Select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import selectinload, joinedload, immediateload
from sqlalchemy import Column, Integer, DateTime, String, Text, ForeignKey, Boolean, Identity, Enum
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from typing import Any, Generator, AsyncGenerator, Iterable, Optional, Union, Type, Dict, cast, TYPE_CHECKING, List, Tuple, TypeVar, Callable
from lazyops.utils import create_unique_id, create_timestamp
from lazyops.utils.logs import logger
from lazyops.types import lazyproperty
from lazyops.libs.psqldb.base import Base, PostgresDB, AsyncSession, Session
from lazyops.libs.psqldb.utils import SQLJson, get_pydantic_model, object_serializer
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
class BaseSQLModel(Base):
    """
    Abstract base class for all SQL models
    """
    __abstract__ = True
    __allow_unmapped__ = True
    id: str = Column(Text, default=create_unique_id, primary_key=True, index=True)
    created_at: datetime.datetime = Column(DateTime(timezone=True), default=create_timestamp, server_default=func.now())
    updated_at: datetime.datetime = Column(DateTime(timezone=True), default=create_timestamp, server_default=func.now(), onupdate=create_timestamp)

    @lazyproperty
    def pydantic_model(self) -> Type[BaseModel]:
        """
        Return the Pydantic model for this ORM model.
        """
        return get_pydantic_model(self)

    @classmethod
    def _build_query(cls, query: Optional[Select]=None, load_attrs: Optional[List[str]]=None, load_attr_method: Optional[Union[str, Callable]]=None, **kwargs) -> Select:
        """
        Build a query
        """
        query = query if query is not None else select(cls)
        query = query.where(*[getattr(cls, key) == value for key, value in kwargs.items()])
        if load_attrs:
            load_attr_method = get_attr_func(load_attr_method)
            for attr in load_attrs:
                query = query.options(load_attr_method(getattr(cls, attr)))
        return query

    def dict(self, exclude: Optional[List[str]]=None, include: Optional[List[str]]=None, safe_encode: Optional[bool]=False, **kwargs) -> Dict[str, Any]:
        """
        Return a dictionary representation of the model.
        """
        data = self.pydantic_model.dict(exclude=exclude, include=include, **kwargs)
        if safe_encode:
            data = {key: object_serializer(value) for key, value in data.items()}
        return data

    def json(self, exclude: Optional[List[str]]=None, include: Optional[List[str]]=None, exclude_none: Optional[bool]=False, **kwargs) -> Dict[str, Any]:
        """
        Return a dictionary representation of the model.
        """
        return self.pydantic_model.json(exclude=exclude, include=include, exclude_none=exclude_none, **kwargs)

    @classmethod
    def _filter(cls, query: Optional[Select]=None, **kwargs) -> Select:
        """
        Build a filter query
        """
        query = query if query is not None else select(cls)
        return query.where(*[getattr(cls, key) == value for key, value in kwargs.items()])

    def _filter_update_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Filter update data
        """
        data = {}
        for field, value in kwargs.items():
            if not hasattr(self, field):
                continue
            with contextlib.suppress(ValueError):
                if hasattr(value, 'all') and hasattr(getattr(self, field), 'all') and (getattr(self, field).all() == value.all()):
                    continue
                if getattr(self, field) == value:
                    continue
            data[field] = value
        return data or None

    @classmethod
    def _handle_exception(cls, msg: Optional[str]=None, error: Optional[Exception]=None, verbose: Optional[bool]=False):
        """
        Handle exception
        """
        msg = msg or f'{cls.__name__} not found'
        if verbose:
            logger.trace(msg, error=error)
        raise error or HTTPException(status_code=404, detail=msg)

    @classmethod
    def session(cls, session: Optional[Session]=None, **kwargs):
        """
        Get session
        """
        return PostgresDB.session(session=session, **kwargs)

    @classmethod
    def async_session(cls, session: Optional[AsyncSession]=None, **kwargs):
        """
        Get async session
        """
        return PostgresDB.async_session(session=session, **kwargs)

    def cast_to(self, model: ModelType) -> ModelType:
        """
        Cast to another model
        """
        return cast(model, self)

    @contextlib.contextmanager
    @classmethod
    def safe_ctx(cls, func: str, default: Optional[Any]=None, _raise_exceptions: Optional[bool]=True, _verbose: Optional[bool]=False, **kwargs):
        """
        Session context manager to handle exceptions
        """
        try:
            yield
        except Exception as e:
            if _raise_exceptions:
                cls._handle_exception(error=e, verbose=_verbose)
            elif _verbose:
                logger.trace(f'Error {func} for {cls.__name__}', error=e)
            return default
        finally:
            pass

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}({self.dict()})>'