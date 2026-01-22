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
class SyncSQLModel(BaseSQLModel):
    """
    Synchronous SQL model
    """
    __abstract__ = True
    __allow_unmapped__ = True

    @classmethod
    def create(cls, session: Optional[Session]=None, _raise_exceptions: Optional[bool]=True, _verbose: Optional[bool]=False, **kwargs) -> SyncSQLModelT:
        """
        Create a record
        """
        with cls.session(session=session) as db_sess:
            try:
                record = cls(**kwargs)
                db_sess.add(record)
                db_sess.commit()
                db_sess.refresh(record)
                return record
            except Exception as e:
                if _raise_exceptions:
                    cls._handle_exception(error=e, verbose=_verbose)
                elif _verbose:
                    logger.trace(f'Error creating {cls.__name__}', error=e)
                return None

    @classmethod
    def get(cls, session: Optional[Session]=None, _load_attrs: Optional[List[str]]=None, _load_attr_method: Optional[Union[str, Callable]]=None, _readonly: Optional[bool]=False, _raise_exceptions: Optional[bool]=True, _verbose: Optional[bool]=False, **kwargs) -> SyncSQLModelT:
        """
        Get a record
        """
        with cls.session(session=session, ro=_readonly) as db_sess:
            query = cls._build_query(load_attrs=_load_attrs, load_attr_method=_load_attr_method, **kwargs)
            result = db_sess.execute(query).scalar_one_or_none()
            if not result and _raise_exceptions:
                cls._handle_exception(error=NoResultFound(), verbose=_verbose)
            return result

    @classmethod
    def get_all(cls, session: Optional[Session]=None, _skip: Optional[int]=None, _limit: Optional[int]=None, _load_attrs: Optional[List[str]]=None, _load_attr_method: Optional[Union[str, Callable]]=None, _readonly: Optional[bool]=False, _raise_exceptions: Optional[bool]=True, _verbose: Optional[bool]=False, **kwargs) -> List[SyncSQLModelT]:
        """
        Get all records
        """
        with cls.session(session=session, ro=_readonly) as db_sess:
            query = cls._build_query(load_attrs=_load_attrs, load_attr_method=_load_attr_method, **kwargs)
            if _skip is not None:
                query = query.offset(_skip)
            if _limit is not None:
                query = query.limit(_limit)
            results = db_sess.scalars(query).all()
            if not results and _raise_exceptions:
                cls._handle_exception(error=NoResultFound(), verbose=_verbose)
            return results

    @classmethod
    def get_first(cls, session: Optional[Session]=None, _load_attrs: Optional[List[str]]=None, _load_attr_method: Optional[Union[str, Callable]]=None, _readonly: Optional[bool]=False, _raise_exceptions: Optional[bool]=True, _verbose: Optional[bool]=False, **kwargs) -> Optional[SyncSQLModelT]:
        """
        Return the first result of a query.
        """
        with cls.session(session=session, ro=_readonly) as db_sess:
            query = cls._build_query(load_attrs=_load_attrs, load_attr_method=_load_attr_method, **kwargs)
            result = db_sess.scalars(query).first()
            if not result and _raise_exceptions:
                cls._handle_exception(error=NoResultFound(), verbose=_verbose)
            return result

    @classmethod
    def get_batch(cls, session: Optional[Session]=None, _skip: Optional[int]=None, _limit: Optional[int]=None, **kwargs) -> Optional[List[SyncSQLModelT]]:
        """
        Returns the existing records
        """
        with cls.session(session=session) as db_sess:
            query = select(cls)
            for key, value in kwargs.items():
                if isinstance(value, (list, tuple, set)):
                    query = query.filter(getattr(cls, key).in_(value))
                else:
                    query = query.filter(getattr(cls, key) == value)
            if _skip is not None:
                query = query.offset(_skip)
            if _limit is not None:
                query = query.limit(_limit)
            return db_sess.execute(query).scalars().all()

    @classmethod
    def delete(cls, session: Optional[Session]=None, _raise_exceptions: Optional[bool]=True, _verbose: Optional[bool]=False, **kwargs) -> Optional[SyncSQLModelT]:
        """
        Delete a record
        """
        with cls.session(session=session) as db_sess:
            obj = cls.get(session=db_sess, _raise_exceptions=_raise_exceptions, _verbose=_verbose, **kwargs)
            if obj is not None:
                db_sess.delete(obj)
                db_sess.commit()
            return obj

    @classmethod
    def delete_batch(cls, session: Optional[Session]=None, _skip: Optional[int]=None, _limit: Optional[int]=None, **kwargs) -> int:
        """
        Delete batch of records
        """
        with cls.session(session=session) as db_sess:
            results = cls.get_batch(session=db_sess, _skip=_skip, _limit=_limit, **kwargs)
            if results:
                with db_sess.begin():
                    for result in results:
                        db_sess.delete(result)
                db_sess.commit()
            return len(results)

    @classmethod
    def delete_all(cls, session: Optional[Session]=None, _raise_exceptions: Optional[bool]=False, _verbose: Optional[bool]=False, **kwargs) -> int:
        """
        Delete all records
        """
        with cls.session(session=session) as db_sess:
            num_records = cls.count(session=db_sess, _raise_exceptions=_raise_exceptions, _verbose=_verbose, **kwargs)
            query = sqlalchemy_delete(cls)
            with cls.safe_ctx('delete all records', default=0, _raise_exceptions=_raise_exceptions, _verbose=_verbose):
                db_sess.execute(query)
                db_sess.commit()
                return num_records

    @classmethod
    def count(cls, session: Optional[Session]=None, _readonly: Optional[bool]=True, _raise_exceptions: Optional[bool]=False, _verbose: Optional[bool]=False, **kwargs) -> int:
        """
        Return the number of records
        """
        with cls.session(session=session, ro=_readonly) as db_sess:
            query = cls._filter(**kwargs) if kwargs else select(func.count(cls.id))
            with cls.safe_ctx('retrieving count', default=0, _raise_exceptions=_raise_exceptions, _verbose=_verbose):
                return db_sess.execute(query).scalar()

    @classmethod
    def exists(cls, session: Optional[Session]=None, _readonly: Optional[bool]=True, _raise_exceptions: Optional[bool]=False, _verbose: Optional[bool]=False, **kwargs) -> bool:
        """
        Return True if the record exists
        """
        with cls.session(session=session, ro=_readonly) as db_sess:
            query = cls._filter(sqlalchemy_exists(), **kwargs).select()
            try:
                return db_sess.execute(query).scalar()
            except Exception as e:
                if _raise_exceptions:
                    cls._handle_exception(error=e, verbose=_verbose)
                elif _verbose:
                    logger.trace(f'Error retrieving exists for {cls.__name__}', error=e)
                return False

    def update(self, session: Optional[Session]=None, _raise_exceptions: Optional[bool]=True, _verbose: Optional[bool]=False, **kwargs) -> SyncSQLModelT:
        """
        Update a record
        """
        with self.session(session=session) as db_sess:
            query = sqlalchemy_update(self.__class__).where(self.__class__.id == self.id).values(**kwargs)
            try:
                db_sess.execute(query)
                db_sess.commit()
            except Exception as e:
                if _raise_exceptions:
                    self._handle_exception(error=e, verbose=_verbose)
                elif _verbose:
                    logger.trace(f'Error updating {self.__class__.__name__}', error=e)
            return self

    def update_inplace(self, session: Optional[Session]=None, _raise_exceptions: Optional[bool]=True, _verbose: Optional[bool]=False, **kwargs) -> SyncSQLModelT:
        """
        Update a record in place
        """
        with self.session(session=session) as db_sess:
            for key, value in kwargs.items():
                setattr(self, key, value)
            try:
                local_object = db_sess.merge(self)
                db_sess.add(local_object)
                db_sess.commit()
                db_sess.flush()
                self = local_object
            except Exception as e:
                if _raise_exceptions:
                    self._handle_exception(error=e, verbose=_verbose)
                elif _verbose:
                    logger.trace(f'Error updating {self.__class__.__name__}', error=e)
            return self

    @classmethod
    def upsert(cls, session: Optional[Session]=None, _raise_exceptions: Optional[bool]=True, _verbose: Optional[bool]=False, **kwargs) -> SyncSQLModelT:
        """
        Upsert a record
        """
        with cls.session(session=session) as db_sess:
            query = sqlalchemy_insert(cls).values(**kwargs)
            query = query.on_conflict_do_update(index_elements=[cls.id], set_=kwargs)
            try:
                db_sess.execute(query)
                db_sess.commit()
            except Exception as e:
                if _raise_exceptions:
                    cls._handle_exception(error=e, verbose=_verbose)
                elif _verbose:
                    logger.trace(f'Error upserting {cls.__name__}', error=e)
            return cls.get(session=db_sess, _raise_exceptions=_raise_exceptions, _verbose=_verbose, **kwargs)

    @classmethod
    def get_or_create(cls, session: Optional[Session]=None, _filterby: Optional[Iterable[str]]=None, _raise_exceptions: Optional[bool]=True, _verbose: Optional[bool]=False, **kwargs) -> Tuple[SyncSQLModelT, bool]:
        """
        Create a record if it doesn't exist

        Returns:
            Tuple[SyncSQLModelT, bool]: The record and a boolean indicating if it was created
        """
        with cls.session(session=session) as db_sess:
            _filterby = [list(kwargs.keys())[0]] if _filterby is None else _filterby
            filterby = {key: kwargs.get(key) for key in _filterby}
            result = cls.get(session=session, _raise_exceptions=False, _verbose=False, **filterby)
            if result is not None:
                return (result, False)
            return (cls.create(session=session, _raise_exceptions=_raise_exceptions, _verbose=_verbose, **kwargs), True)

    @classmethod
    def get_or_none(cls, session: Optional[Session]=None, **kwargs) -> Optional[SyncSQLModelT]:
        """
        Get a record or None if it doesn't exist
        """
        return cls.get(session=session, _raise_exceptions=False, _verbose=False, **kwargs)