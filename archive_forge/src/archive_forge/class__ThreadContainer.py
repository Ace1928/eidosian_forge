from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
class _ThreadContainer(object):
    """
    Encapsulates the capability to contain Thread objects.

    @group Instrumentation:
        start_thread

    @group Threads snapshot:
        scan_threads,
        get_thread, get_thread_count, get_thread_ids,
        has_thread, iter_threads, iter_thread_ids,
        find_threads_by_name, get_windows,
        clear_threads, clear_dead_threads, close_thread_handles
    """

    def __init__(self):
        self.__threadDict = dict()

    def __initialize_snapshot(self):
        """
        Private method to automatically initialize the snapshot
        when you try to use it without calling any of the scan_*
        methods first. You don't need to call this yourself.
        """
        if not self.__threadDict:
            self.scan_threads()

    def __contains__(self, anObject):
        """
        @type  anObject: L{Thread}, int
        @param anObject:
             - C{int}: Global ID of the thread to look for.
             - C{Thread}: Thread object to look for.

        @rtype:  bool
        @return: C{True} if the snapshot contains
            a L{Thread} object with the same ID.
        """
        if isinstance(anObject, Thread):
            anObject = anObject.dwThreadId
        return self.has_thread(anObject)

    def __iter__(self):
        """
        @see:    L{iter_threads}
        @rtype:  dictionary-valueiterator
        @return: Iterator of L{Thread} objects in this snapshot.
        """
        return self.iter_threads()

    def __len__(self):
        """
        @see:    L{get_thread_count}
        @rtype:  int
        @return: Count of L{Thread} objects in this snapshot.
        """
        return self.get_thread_count()

    def has_thread(self, dwThreadId):
        """
        @type  dwThreadId: int
        @param dwThreadId: Global ID of the thread to look for.

        @rtype:  bool
        @return: C{True} if the snapshot contains a
            L{Thread} object with the given global ID.
        """
        self.__initialize_snapshot()
        return dwThreadId in self.__threadDict

    def get_thread(self, dwThreadId):
        """
        @type  dwThreadId: int
        @param dwThreadId: Global ID of the thread to look for.

        @rtype:  L{Thread}
        @return: Thread object with the given global ID.
        """
        self.__initialize_snapshot()
        if dwThreadId not in self.__threadDict:
            msg = 'Unknown thread ID: %d' % dwThreadId
            raise KeyError(msg)
        return self.__threadDict[dwThreadId]

    def iter_thread_ids(self):
        """
        @see:    L{iter_threads}
        @rtype:  dictionary-keyiterator
        @return: Iterator of global thread IDs in this snapshot.
        """
        self.__initialize_snapshot()
        return compat.iterkeys(self.__threadDict)

    def iter_threads(self):
        """
        @see:    L{iter_thread_ids}
        @rtype:  dictionary-valueiterator
        @return: Iterator of L{Thread} objects in this snapshot.
        """
        self.__initialize_snapshot()
        return compat.itervalues(self.__threadDict)

    def get_thread_ids(self):
        """
        @rtype:  list( int )
        @return: List of global thread IDs in this snapshot.
        """
        self.__initialize_snapshot()
        return compat.keys(self.__threadDict)

    def get_thread_count(self):
        """
        @rtype:  int
        @return: Count of L{Thread} objects in this snapshot.
        """
        self.__initialize_snapshot()
        return len(self.__threadDict)

    def find_threads_by_name(self, name, bExactMatch=True):
        """
        Find threads by name, using different search methods.

        @type  name: str, None
        @param name: Name to look for. Use C{None} to find nameless threads.

        @type  bExactMatch: bool
        @param bExactMatch: C{True} if the name must be
            B{exactly} as given, C{False} if the name can be
            loosely matched.

            This parameter is ignored when C{name} is C{None}.

        @rtype:  list( L{Thread} )
        @return: All threads matching the given name.
        """
        found_threads = list()
        if name is None:
            for aThread in self.iter_threads():
                if aThread.get_name() is None:
                    found_threads.append(aThread)
        elif bExactMatch:
            for aThread in self.iter_threads():
                if aThread.get_name() == name:
                    found_threads.append(aThread)
        else:
            for aThread in self.iter_threads():
                t_name = aThread.get_name()
                if t_name is not None and name in t_name:
                    found_threads.append(aThread)
        return found_threads

    def get_windows(self):
        """
        @rtype:  list of L{Window}
        @return: Returns a list of windows handled by this process.
        """
        window_list = list()
        for thread in self.iter_threads():
            window_list.extend(thread.get_windows())
        return window_list

    def start_thread(self, lpStartAddress, lpParameter=0, bSuspended=False):
        """
        Remotely creates a new thread in the process.

        @type  lpStartAddress: int
        @param lpStartAddress: Start address for the new thread.

        @type  lpParameter: int
        @param lpParameter: Optional argument for the new thread.

        @type  bSuspended: bool
        @param bSuspended: C{True} if the new thread should be suspended.
            In that case use L{Thread.resume} to start execution.
        """
        if bSuspended:
            dwCreationFlags = win32.CREATE_SUSPENDED
        else:
            dwCreationFlags = 0
        hProcess = self.get_handle(win32.PROCESS_CREATE_THREAD | win32.PROCESS_QUERY_INFORMATION | win32.PROCESS_VM_OPERATION | win32.PROCESS_VM_WRITE | win32.PROCESS_VM_READ)
        hThread, dwThreadId = win32.CreateRemoteThread(hProcess, 0, 0, lpStartAddress, lpParameter, dwCreationFlags)
        aThread = Thread(dwThreadId, hThread, self)
        self._add_thread(aThread)
        return aThread

    def scan_threads(self):
        """
        Populates the snapshot with running threads.
        """
        dwProcessId = self.get_pid()
        if dwProcessId in (0, 4, 8):
            return
        dead_tids = self._get_thread_ids()
        dwProcessId = self.get_pid()
        hSnapshot = win32.CreateToolhelp32Snapshot(win32.TH32CS_SNAPTHREAD, dwProcessId)
        try:
            te = win32.Thread32First(hSnapshot)
            while te is not None:
                if te.th32OwnerProcessID == dwProcessId:
                    dwThreadId = te.th32ThreadID
                    if dwThreadId in dead_tids:
                        dead_tids.remove(dwThreadId)
                    if not self._has_thread_id(dwThreadId):
                        aThread = Thread(dwThreadId, process=self)
                        self._add_thread(aThread)
                te = win32.Thread32Next(hSnapshot)
        finally:
            win32.CloseHandle(hSnapshot)
        for tid in dead_tids:
            self._del_thread(tid)

    def clear_dead_threads(self):
        """
        Remove Thread objects from the snapshot
        referring to threads no longer running.
        """
        for tid in self.get_thread_ids():
            aThread = self.get_thread(tid)
            if not aThread.is_alive():
                self._del_thread(aThread)

    def clear_threads(self):
        """
        Clears the threads snapshot.
        """
        for aThread in compat.itervalues(self.__threadDict):
            aThread.clear()
        self.__threadDict = dict()

    def close_thread_handles(self):
        """
        Closes all open handles to threads in the snapshot.
        """
        for aThread in self.iter_threads():
            try:
                aThread.close_handle()
            except Exception:
                try:
                    e = sys.exc_info()[1]
                    msg = 'Cannot close thread handle %s, reason: %s'
                    msg %= (aThread.hThread.value, str(e))
                    warnings.warn(msg)
                except Exception:
                    pass

    def _add_thread(self, aThread):
        """
        Private method to add a thread object to the snapshot.

        @type  aThread: L{Thread}
        @param aThread: Thread object.
        """
        dwThreadId = aThread.dwThreadId
        aThread.set_process(self)
        self.__threadDict[dwThreadId] = aThread

    def _del_thread(self, dwThreadId):
        """
        Private method to remove a thread object from the snapshot.

        @type  dwThreadId: int
        @param dwThreadId: Global thread ID.
        """
        try:
            aThread = self.__threadDict[dwThreadId]
            del self.__threadDict[dwThreadId]
        except KeyError:
            aThread = None
            msg = 'Unknown thread ID %d' % dwThreadId
            warnings.warn(msg, RuntimeWarning)
        if aThread:
            aThread.clear()

    def _has_thread_id(self, dwThreadId):
        """
        Private method to test for a thread in the snapshot without triggering
        an automatic scan.
        """
        return dwThreadId in self.__threadDict

    def _get_thread_ids(self):
        """
        Private method to get the list of thread IDs currently in the snapshot
        without triggering an automatic scan.
        """
        return compat.keys(self.__threadDict)

    def __add_created_thread(self, event):
        """
        Private method to automatically add new thread objects from debug events.

        @type  event: L{Event}
        @param event: Event object.
        """
        dwThreadId = event.get_tid()
        hThread = event.get_thread_handle()
        if not self._has_thread_id(dwThreadId):
            aThread = Thread(dwThreadId, hThread, self)
            teb_ptr = event.get_teb()
            if teb_ptr:
                aThread._teb_ptr = teb_ptr
            self._add_thread(aThread)

    def _notify_create_process(self, event):
        """
        Notify the creation of the main thread of this process.

        This is done automatically by the L{Debug} class, you shouldn't need
        to call it yourself.

        @type  event: L{CreateProcessEvent}
        @param event: Create process event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
        self.__add_created_thread(event)
        return True

    def _notify_create_thread(self, event):
        """
        Notify the creation of a new thread in this process.

        This is done automatically by the L{Debug} class, you shouldn't need
        to call it yourself.

        @type  event: L{CreateThreadEvent}
        @param event: Create thread event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
        self.__add_created_thread(event)
        return True

    def _notify_exit_thread(self, event):
        """
        Notify the termination of a thread.

        This is done automatically by the L{Debug} class, you shouldn't need
        to call it yourself.

        @type  event: L{ExitThreadEvent}
        @param event: Exit thread event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
        dwThreadId = event.get_tid()
        if self._has_thread_id(dwThreadId):
            self._del_thread(dwThreadId)
        return True