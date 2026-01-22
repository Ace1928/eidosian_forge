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
class CrashDTO(BaseDTO):
    """
    Database mapping for crash dumps.
    """
    __tablename__ = 'crashes'
    id = Column(Integer, Sequence(__tablename__ + '_seq'), primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    exploitable = Column(Integer, nullable=False)
    exploitability_rule = Column(String(32), nullable=False)
    exploitability_rating = Column(String(32), nullable=False)
    exploitability_desc = Column(String, nullable=False)
    os = Column(String(32), nullable=False)
    arch = Column(String(16), nullable=False)
    bits = Column(Integer, nullable=False)
    event = Column(String, nullable=False)
    pid = Column(Integer, nullable=False)
    tid = Column(Integer, nullable=False)
    pc = Column(BigInteger, nullable=False)
    sp = Column(BigInteger, nullable=False)
    fp = Column(BigInteger, nullable=False)
    pc_label = Column(String, nullable=False)
    exception = Column(String(64))
    exception_text = Column(String(64))
    exception_address = Column(BigInteger)
    exception_label = Column(String)
    first_chance = Column(Boolean)
    fault_type = Column(Integer)
    fault_address = Column(BigInteger)
    fault_label = Column(String)
    fault_disasm = Column(String)
    stack_trace = Column(String)
    command_line = Column(String)
    environment = Column(String)
    debug_string = Column(String)
    notes = Column(String)
    signature = Column(String, nullable=False)
    data = deferred(Column(LargeBinary, nullable=False))

    def __init__(self, crash):
        """
        @type  crash: Crash
        @param crash: L{Crash} object to store into the database.
        """
        self.timestamp = datetime.datetime.fromtimestamp(crash.timeStamp)
        self.signature = pickle.dumps(crash.signature, protocol=0)
        memoryMap = crash.memoryMap
        try:
            crash.memoryMap = None
            self.data = buffer(Marshaller.dumps(crash))
        finally:
            crash.memoryMap = memoryMap
        self.exploitability_rating, self.exploitability_rule, self.exploitability_desc = crash.isExploitable()
        self.exploitable = ['Not an exception', 'Not exploitable', 'Not likely exploitable', 'Unknown', 'Probably exploitable', 'Exploitable'].index(self.exploitability_rating)
        self.os = crash.os
        self.arch = crash.arch
        self.bits = crash.bits
        self.event = crash.eventName
        self.pid = crash.pid
        self.tid = crash.tid
        self.pc = crash.pc
        self.sp = crash.sp
        self.fp = crash.fp
        self.pc_label = crash.labelPC
        self.exception = crash.exceptionName
        self.exception_text = crash.exceptionDescription
        self.exception_address = crash.exceptionAddress
        self.exception_label = crash.exceptionLabel
        self.first_chance = crash.firstChance
        self.fault_type = crash.faultType
        self.fault_address = crash.faultAddress
        self.fault_label = crash.faultLabel
        self.fault_disasm = CrashDump.dump_code(crash.faultDisasm, crash.pc)
        self.stack_trace = CrashDump.dump_stack_trace_with_labels(crash.stackTracePretty)
        self.command_line = crash.commandLine
        if crash.environment:
            envList = crash.environment.items()
            envList.sort()
            environment = ''
            for envKey, envVal in envList:
                environment += envKey + '=' + envVal + '\n'
            if environment:
                self.environment = environment
        self.debug_string = crash.debugString
        self.notes = crash.notesReport()

    def toCrash(self, getMemoryDump=False):
        """
        Returns a L{Crash} object using the data retrieved from the database.

        @type  getMemoryDump: bool
        @param getMemoryDump: If C{True} retrieve the memory dump.
            Defaults to C{False} since this may be a costly operation.

        @rtype:  L{Crash}
        @return: Crash object.
        """
        crash = Marshaller.loads(str(self.data))
        if not isinstance(crash, Crash):
            raise TypeError('Expected Crash instance, got %s instead' % type(crash))
        crash._rowid = self.id
        if not crash.memoryMap:
            memory = getattr(self, 'memory', [])
            if memory:
                crash.memoryMap = [dto.toMBI(getMemoryDump) for dto in memory]
        return crash