import sqlite3
import datetime
import warnings
from sqlalchemy import create_engine, Column, ForeignKey, Sequence
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.interfaces import PoolListener
from sqlalchemy.orm import sessionmaker, deferred
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound
from sqlalchemy.types import Integer, BigInteger, Boolean, DateTime, String, \
from sqlalchemy.sql.expression import asc, desc
from crash import Crash, Marshaller, pickle, HIGHEST_PROTOCOL
from textio import CrashDump
import win32
class CrashDAO(BaseDAO):
    """
    Data Access Object to read, write and search for L{Crash} objects in a
    database.
    """

    @Transactional
    def add(self, crash, allow_duplicates=True):
        """
        Add a new crash dump to the database, optionally filtering them by
        signature to avoid duplicates.

        @type  crash: L{Crash}
        @param crash: Crash object.

        @type  allow_duplicates: bool
        @param allow_duplicates: (Optional)
            C{True} to always add the new crash dump.
            C{False} to only add the crash dump if no other crash with the
            same signature is found in the database.

            Sometimes, your fuzzer turns out to be I{too} good. Then you find
            youself browsing through gigabytes of crash dumps, only to find
            a handful of actual bugs in them. This simple heuristic filter
            saves you the trouble by discarding crashes that seem to be similar
            to another one you've already found.
        """
        if not allow_duplicates:
            signature = pickle.dumps(crash.signature, protocol=0)
            if self._session.query(CrashDTO.id).filter_by(signature=signature).count() > 0:
                return
        crash_id = self.__add_crash(crash)
        self.__add_memory(crash_id, crash.memoryMap)
        crash._rowid = crash_id

    def __add_crash(self, crash):
        session = self._session
        r_crash = None
        try:
            r_crash = CrashDTO(crash)
            session.add(r_crash)
            session.flush()
            crash_id = r_crash.id
        finally:
            try:
                if r_crash is not None:
                    session.expire(r_crash)
            finally:
                del r_crash
        return crash_id

    def __add_memory(self, crash_id, memoryMap):
        session = self._session
        if memoryMap:
            for mbi in memoryMap:
                r_mem = MemoryDTO(crash_id, mbi)
                session.add(r_mem)
                session.flush()

    @Transactional
    def find(self, signature=None, order=0, since=None, until=None, offset=None, limit=None):
        """
        Retrieve all crash dumps in the database, optionally filtering them by
        signature and timestamp, and/or sorting them by timestamp.

        Results can be paged to avoid consuming too much memory if the database
        is large.

        @see: L{find_by_example}

        @type  signature: object
        @param signature: (Optional) Return only through crashes matching
            this signature. See L{Crash.signature} for more details.

        @type  order: int
        @param order: (Optional) Sort by timestamp.
            If C{== 0}, results are not sorted.
            If C{> 0}, results are sorted from older to newer.
            If C{< 0}, results are sorted from newer to older.

        @type  since: datetime
        @param since: (Optional) Return only the crashes after and
            including this date and time.

        @type  until: datetime
        @param until: (Optional) Return only the crashes before this date
            and time, not including it.

        @type  offset: int
        @param offset: (Optional) Skip the first I{offset} results.

        @type  limit: int
        @param limit: (Optional) Return at most I{limit} results.

        @rtype:  list(L{Crash})
        @return: List of Crash objects.
        """
        if since and until and (since > until):
            warnings.warn("CrashDAO.find() got the 'since' and 'until' arguments reversed, corrected automatically.")
            since, until = (until, since)
        if limit is not None and (not limit):
            warnings.warn('CrashDAO.find() was set a limit of 0 results, returning without executing a query.')
            return []
        query = self._session.query(CrashDTO)
        if signature is not None:
            sig_pickled = pickle.dumps(signature, protocol=0)
            query = query.filter(CrashDTO.signature == sig_pickled)
        if since:
            query = query.filter(CrashDTO.timestamp >= since)
        if until:
            query = query.filter(CrashDTO.timestamp < until)
        if order:
            if order > 0:
                query = query.order_by(asc(CrashDTO.timestamp))
            else:
                query = query.order_by(desc(CrashDTO.timestamp))
        else:
            query = query.order_by(asc(CrashDTO.id))
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)
        try:
            return [dto.toCrash() for dto in query.all()]
        except NoResultFound:
            return []

    @Transactional
    def find_by_example(self, crash, offset=None, limit=None):
        """
        Find all crash dumps that have common properties with the crash dump
        provided.

        Results can be paged to avoid consuming too much memory if the database
        is large.

        @see: L{find}

        @type  crash: L{Crash}
        @param crash: Crash object to compare with. Fields set to C{None} are
            ignored, all other fields but the signature are used in the
            comparison.

            To search for signature instead use the L{find} method.

        @type  offset: int
        @param offset: (Optional) Skip the first I{offset} results.

        @type  limit: int
        @param limit: (Optional) Return at most I{limit} results.

        @rtype:  list(L{Crash})
        @return: List of similar crash dumps found.
        """
        if limit is not None and (not limit):
            warnings.warn('CrashDAO.find_by_example() was set a limit of 0 results, returning without executing a query.')
            return []
        query = self._session.query(CrashDTO)
        query = query.asc(CrashDTO.id)
        dto = CrashDTO(crash)
        for name, column in compat.iteritems(CrashDTO.__dict__):
            if not name.startswith('__') and name not in ('id', 'signature', 'data'):
                if isinstance(column, Column):
                    value = getattr(dto, name, None)
                    if value is not None:
                        query = query.filter(column == value)
        if offset:
            query = query.offset(offset)
        if limit:
            query = query.limit(limit)
        try:
            return [dto.toCrash() for dto in query.all()]
        except NoResultFound:
            return []

    @Transactional
    def count(self, signature=None):
        """
        Counts how many crash dumps have been stored in this database.
        Optionally filters the count by heuristic signature.

        @type  signature: object
        @param signature: (Optional) Count only the crashes that match
            this signature. See L{Crash.signature} for more details.

        @rtype:  int
        @return: Count of crash dumps stored in this database.
        """
        query = self._session.query(CrashDTO.id)
        if signature:
            sig_pickled = pickle.dumps(signature, protocol=0)
            query = query.filter_by(signature=sig_pickled)
        return query.count()

    @Transactional
    def delete(self, crash):
        """
        Remove the given crash dump from the database.

        @type  crash: L{Crash}
        @param crash: Crash dump to remove.
        """
        query = self._session.query(CrashDTO).filter_by(id=crash._rowid)
        query.delete(synchronize_session=False)
        del crash._rowid