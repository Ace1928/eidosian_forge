from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.textio import HexDump, CrashDump
from winappdbg.util import StaticClass, MemoryAddresses, PathOperations
import sys
import os
import time
import zlib
import warnings
class CrashContainer(object):
    """
    Old crash dump persistencer using a DBM database.
    Doesn't support duplicate crashes.

    @warning:
        DBM database support is provided for backwards compatibility with older
        versions of WinAppDbg. New applications should not use this class.
        Also, DBM databases in Python suffer from multiple problems that can
        easily be avoided by switching to a SQL database.

    @see: If you really must use a DBM database, try the standard C{shelve}
        module instead: U{http://docs.python.org/library/shelve.html}

    @group Marshalling configuration:
        optimizeKeys, optimizeValues, compressKeys, compressValues, escapeKeys,
        escapeValues, binaryKeys, binaryValues

    @type optimizeKeys: bool
    @cvar optimizeKeys: Ignored by the current implementation.

        Up to WinAppDbg 1.4 this setting caused the database keys to be
        optimized when pickled with the standard C{pickle} module.

        But with a DBM database backend that causes inconsistencies, since the
        same key can be serialized into multiple optimized pickles, thus losing
        uniqueness.

    @type optimizeValues: bool
    @cvar optimizeValues: C{True} to optimize the marshalling of keys, C{False}
        otherwise. Only used with the C{pickle} module, ignored when using the
        more secure C{cerealizer} module.

    @type compressKeys: bool
    @cvar compressKeys: C{True} to compress keys when marshalling, C{False}
        to leave them uncompressed.

    @type compressValues: bool
    @cvar compressValues: C{True} to compress values when marshalling, C{False}
        to leave them uncompressed.

    @type escapeKeys: bool
    @cvar escapeKeys: C{True} to escape keys when marshalling, C{False}
        to leave them uncompressed.

    @type escapeValues: bool
    @cvar escapeValues: C{True} to escape values when marshalling, C{False}
        to leave them uncompressed.

    @type binaryKeys: bool
    @cvar binaryKeys: C{True} to marshall keys to binary format (the Python
        C{buffer} type), C{False} to use text marshalled keys (C{str} type).

    @type binaryValues: bool
    @cvar binaryValues: C{True} to marshall values to binary format (the Python
        C{buffer} type), C{False} to use text marshalled values (C{str} type).
    """
    optimizeKeys = False
    optimizeValues = True
    compressKeys = False
    compressValues = True
    escapeKeys = False
    escapeValues = False
    binaryKeys = False
    binaryValues = False

    def __init__(self, filename=None, allowRepeatedKeys=False):
        """
        @type  filename: str
        @param filename: (Optional) File name for crash database.
            If no filename is specified, the container is volatile.

            Volatile containers are stored only in memory and
            destroyed when they go out of scope.

        @type  allowRepeatedKeys: bool
        @param allowRepeatedKeys:
            Currently not supported, always use C{False}.
        """
        if allowRepeatedKeys:
            raise NotImplementedError()
        self.__filename = filename
        if filename:
            global anydbm
            if not anydbm:
                import anydbm
            self.__db = anydbm.open(filename, 'c')
            self.__keys = dict([(self.unmarshall_key(mk), mk) for mk in self.__db.keys()])
        else:
            self.__db = dict()
            self.__keys = dict()

    def remove_key(self, key):
        """
        Removes the given key from the set of known keys.

        @type  key: L{Crash} key.
        @param key: Key to remove.
        """
        del self.__keys[key]

    def marshall_key(self, key):
        """
        Marshalls a Crash key to be used in the database.

        @see: L{__init__}

        @type  key: L{Crash} key.
        @param key: Key to convert.

        @rtype:  str or buffer
        @return: Converted key.
        """
        if key in self.__keys:
            return self.__keys[key]
        skey = pickle.dumps(key, protocol=0)
        if self.compressKeys:
            skey = zlib.compress(skey, zlib.Z_BEST_COMPRESSION)
        if self.escapeKeys:
            skey = skey.encode('hex')
        if self.binaryKeys:
            skey = buffer(skey)
        self.__keys[key] = skey
        return skey

    def unmarshall_key(self, key):
        """
        Unmarshalls a Crash key read from the database.

        @type  key: str or buffer
        @param key: Key to convert.

        @rtype:  L{Crash} key.
        @return: Converted key.
        """
        key = str(key)
        if self.escapeKeys:
            key = key.decode('hex')
        if self.compressKeys:
            key = zlib.decompress(key)
        key = pickle.loads(key)
        return key

    def marshall_value(self, value, storeMemoryMap=False):
        """
        Marshalls a Crash object to be used in the database.
        By default the C{memoryMap} member is B{NOT} stored here.

        @warning: Setting the C{storeMemoryMap} argument to C{True} can lead to
            a severe performance penalty!

        @type  value: L{Crash}
        @param value: Object to convert.

        @type  storeMemoryMap: bool
        @param storeMemoryMap: C{True} to store the memory map, C{False}
            otherwise.

        @rtype:  str
        @return: Converted object.
        """
        if hasattr(value, 'memoryMap'):
            crash = value
            memoryMap = crash.memoryMap
            try:
                crash.memoryMap = None
                if storeMemoryMap and memoryMap is not None:
                    crash.memoryMap = list(memoryMap)
                if self.optimizeValues:
                    value = pickle.dumps(crash, protocol=HIGHEST_PROTOCOL)
                    value = optimize(value)
                else:
                    value = pickle.dumps(crash, protocol=0)
            finally:
                crash.memoryMap = memoryMap
                del memoryMap
                del crash
        if self.compressValues:
            value = zlib.compress(value, zlib.Z_BEST_COMPRESSION)
        if self.escapeValues:
            value = value.encode('hex')
        if self.binaryValues:
            value = buffer(value)
        return value

    def unmarshall_value(self, value):
        """
        Unmarshalls a Crash object read from the database.

        @type  value: str
        @param value: Object to convert.

        @rtype:  L{Crash}
        @return: Converted object.
        """
        value = str(value)
        if self.escapeValues:
            value = value.decode('hex')
        if self.compressValues:
            value = zlib.decompress(value)
        value = pickle.loads(value)
        return value

    def __len__(self):
        """
        @rtype:  int
        @return: Count of known keys.
        """
        return len(self.__keys)

    def __bool__(self):
        """
        @rtype:  bool
        @return: C{False} if there are no known keys.
        """
        return bool(self.__keys)

    def __contains__(self, crash):
        """
        @type  crash: L{Crash}
        @param crash: Crash object.

        @rtype:  bool
        @return:
            C{True} if a Crash object with the same key is in the container.
        """
        return self.has_key(crash.key())

    def has_key(self, key):
        """
        @type  key: L{Crash} key.
        @param key: Key to find.

        @rtype:  bool
        @return: C{True} if the key is present in the set of known keys.
        """
        return key in self.__keys

    def iterkeys(self):
        """
        @rtype:  iterator
        @return: Iterator of known L{Crash} keys.
        """
        return compat.iterkeys(self.__keys)

    class __CrashContainerIterator(object):
        """
        Iterator of Crash objects. Returned by L{CrashContainer.__iter__}.
        """

        def __init__(self, container):
            """
            @type  container: L{CrashContainer}
            @param container: Crash set to iterate.
            """
            self.__container = container
            self.__keys_iter = compat.iterkeys(container)

        def next(self):
            """
            @rtype:  L{Crash}
            @return: A B{copy} of a Crash object in the L{CrashContainer}.
            @raise StopIteration: No more items left.
            """
            key = self.__keys_iter.next()
            return self.__container.get(key)

    def __del__(self):
        """Class destructor. Closes the database when this object is destroyed."""
        try:
            if self.__filename:
                self.__db.close()
        except:
            pass

    def __iter__(self):
        """
        @see:    L{itervalues}
        @rtype:  iterator
        @return: Iterator of the contained L{Crash} objects.
        """
        return self.itervalues()

    def itervalues(self):
        """
        @rtype:  iterator
        @return: Iterator of the contained L{Crash} objects.

        @warning: A B{copy} of each object is returned,
            so any changes made to them will be lost.

            To preserve changes do the following:
                1. Keep a reference to the object.
                2. Delete the object from the set.
                3. Modify the object and add it again.
        """
        return self.__CrashContainerIterator(self)

    def add(self, crash):
        """
        Adds a new crash to the container.
        If the crash appears to be already known, it's ignored.

        @see: L{Crash.key}

        @type  crash: L{Crash}
        @param crash: Crash object to add.
        """
        if crash not in self:
            key = crash.key()
            skey = self.marshall_key(key)
            data = self.marshall_value(crash, storeMemoryMap=True)
            self.__db[skey] = data

    def __delitem__(self, key):
        """
        Removes a crash from the container.

        @type  key: L{Crash} unique key.
        @param key: Key of the crash to get.
        """
        skey = self.marshall_key(key)
        del self.__db[skey]
        self.remove_key(key)

    def remove(self, crash):
        """
        Removes a crash from the container.

        @type  crash: L{Crash}
        @param crash: Crash object to remove.
        """
        del self[crash.key()]

    def get(self, key):
        """
        Retrieves a crash from the container.

        @type  key: L{Crash} unique key.
        @param key: Key of the crash to get.

        @rtype:  L{Crash} object.
        @return: Crash matching the given key.

        @see:     L{iterkeys}
        @warning: A B{copy} of each object is returned,
            so any changes made to them will be lost.

            To preserve changes do the following:
                1. Keep a reference to the object.
                2. Delete the object from the set.
                3. Modify the object and add it again.
        """
        skey = self.marshall_key(key)
        data = self.__db[skey]
        crash = self.unmarshall_value(data)
        return crash

    def __getitem__(self, key):
        """
        Retrieves a crash from the container.

        @type  key: L{Crash} unique key.
        @param key: Key of the crash to get.

        @rtype:  L{Crash} object.
        @return: Crash matching the given key.

        @see:     L{iterkeys}
        @warning: A B{copy} of each object is returned,
            so any changes made to them will be lost.

            To preserve changes do the following:
                1. Keep a reference to the object.
                2. Delete the object from the set.
                3. Modify the object and add it again.
        """
        return self.get(key)