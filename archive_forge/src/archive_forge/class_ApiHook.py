from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
class ApiHook(object):
    """
    Used by L{EventHandler}.

    This class acts as an action callback for code breakpoints set at the
    beginning of a function. It automatically retrieves the parameters from
    the stack, sets a breakpoint at the return address and retrieves the
    return value from the function call.

    @see: L{EventHandler.apiHooks}

    @type modName: str
    @ivar modName: Module name.

    @type procName: str
    @ivar procName: Procedure name.
    """

    def __init__(self, eventHandler, modName, procName, paramCount=None, signature=None):
        """
        @type  eventHandler: L{EventHandler}
        @param eventHandler: Event handler instance. This is where the hook
            callbacks are to be defined (see below).

        @type  modName: str
        @param modName: Module name.

        @type  procName: str
        @param procName: Procedure name.
            The pre and post callbacks will be deduced from it.

            For example, if the procedure is "LoadLibraryEx" the callback
            routines will be "pre_LoadLibraryEx" and "post_LoadLibraryEx".

            The signature for the callbacks should be something like this::

                def pre_LoadLibraryEx(self, event, ra, lpFilename, hFile, dwFlags):

                    # return address
                    ra = params[0]

                    # function arguments start from here...
                    szFilename = event.get_process().peek_string(lpFilename)

                    # (...)

                def post_LoadLibraryEx(self, event, return_value):

                    # (...)

            Note that all pointer types are treated like void pointers, so your
            callback won't get the string or structure pointed to by it, but
            the remote memory address instead. This is so to prevent the ctypes
            library from being "too helpful" and trying to dereference the
            pointer. To get the actual data being pointed to, use one of the
            L{Process.read} methods.

        @type  paramCount: int
        @param paramCount:
            (Optional) Number of parameters for the C{preCB} callback,
            not counting the return address. Parameters are read from
            the stack and assumed to be DWORDs in 32 bits and QWORDs in 64.

            This is a faster way to pull stack parameters in 32 bits, but in 64
            bits (or with some odd APIs in 32 bits) it won't be useful, since
            not all arguments to the hooked function will be of the same size.

            For a more reliable and cross-platform way of hooking use the
            C{signature} argument instead.

        @type  signature: tuple
        @param signature:
            (Optional) Tuple of C{ctypes} data types that constitute the
            hooked function signature. When the function is called, this will
            be used to parse the arguments from the stack. Overrides the
            C{paramCount} argument.
        """
        self.__modName = modName
        self.__procName = procName
        self.__paramCount = paramCount
        self.__signature = signature
        self.__preCB = getattr(eventHandler, 'pre_%s' % procName, None)
        self.__postCB = getattr(eventHandler, 'post_%s' % procName, None)
        self.__hook = dict()

    def __call__(self, event):
        """
        Handles the breakpoint event on entry of the function.

        @type  event: L{ExceptionEvent}
        @param event: Breakpoint hit event.

        @raise WindowsError: An error occured.
        """
        pid = event.get_pid()
        try:
            hook = self.__hook[pid]
        except KeyError:
            hook = Hook(self.__preCB, self.__postCB, self.__paramCount, self.__signature, event.get_process().get_arch())
            self.__hook[pid] = hook
        return hook(event)

    @property
    def modName(self):
        return self.__modName

    @property
    def procName(self):
        return self.__procName

    def hook(self, debug, pid):
        """
        Installs the API hook on a given process and module.

        @warning: Do not call from an API hook callback.

        @type  debug: L{Debug}
        @param debug: Debug object.

        @type  pid: int
        @param pid: Process ID.
        """
        label = '%s!%s' % (self.__modName, self.__procName)
        try:
            hook = self.__hook[pid]
        except KeyError:
            try:
                aProcess = debug.system.get_process(pid)
            except KeyError:
                aProcess = Process(pid)
            hook = Hook(self.__preCB, self.__postCB, self.__paramCount, self.__signature, aProcess.get_arch())
            self.__hook[pid] = hook
        hook.hook(debug, pid, label)

    def unhook(self, debug, pid):
        """
        Removes the API hook from the given process and module.

        @warning: Do not call from an API hook callback.

        @type  debug: L{Debug}
        @param debug: Debug object.

        @type  pid: int
        @param pid: Process ID.
        """
        try:
            hook = self.__hook[pid]
        except KeyError:
            return
        label = '%s!%s' % (self.__modName, self.__procName)
        hook.unhook(debug, pid, label)
        del self.__hook[pid]