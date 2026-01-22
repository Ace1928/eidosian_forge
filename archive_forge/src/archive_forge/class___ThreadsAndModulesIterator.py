from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump, HexInput
from winappdbg.util import Regenerator, PathOperations, MemoryAddresses
from winappdbg.module import Module, _ModuleContainer
from winappdbg.thread import Thread, _ThreadContainer
from winappdbg.window import Window
from winappdbg.search import Search, \
from winappdbg.disasm import Disassembler
import re
import os
import os.path
import ctypes
import struct
import warnings
import traceback
class __ThreadsAndModulesIterator(object):
    """
        Iterator object for L{Process} objects.
        Iterates through L{Thread} objects first, L{Module} objects next.
        """

    def __init__(self, container):
        """
            @type  container: L{Process}
            @param container: L{Thread} and L{Module} container.
            """
        self.__container = container
        self.__iterator = None
        self.__state = 0

    def __iter__(self):
        """x.__iter__() <==> iter(x)"""
        return self

    def next(self):
        """x.next() -> the next value, or raise StopIteration"""
        if self.__state == 0:
            self.__iterator = self.__container.iter_threads()
            self.__state = 1
        if self.__state == 1:
            try:
                return self.__iterator.next()
            except StopIteration:
                self.__iterator = self.__container.iter_modules()
                self.__state = 2
        if self.__state == 2:
            try:
                return self.__iterator.next()
            except StopIteration:
                self.__iterator = None
                self.__state = 3
        raise StopIteration