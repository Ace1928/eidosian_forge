from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __set_break(self, pid, address, action, oneshot):
    """
        Used by L{break_at} and L{stalk_at}.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int or str
        @param address:
            Memory address of code instruction to break at. It can be an
            integer value for the actual address or a string with a label
            to be resolved.

        @type  action: function
        @param action: (Optional) Action callback function.

            See L{define_code_breakpoint} for more details.

        @type  oneshot: bool
        @param oneshot: C{True} for one-shot breakpoints, C{False} otherwise.

        @rtype:  L{Breakpoint}
        @return: Returns the new L{Breakpoint} object, or C{None} if the label
            couldn't be resolved and the breakpoint was deferred. Deferred
            breakpoints are set when the DLL they point to is loaded.
        """
    if type(address) not in (int, long):
        label = address
        try:
            address = self.system.get_process(pid).resolve_label(address)
            if not address:
                raise Exception()
        except Exception:
            try:
                deferred = self.__deferredBP[pid]
            except KeyError:
                deferred = dict()
                self.__deferredBP[pid] = deferred
            if label in deferred:
                msg = 'Redefined deferred code breakpoint at %s in process ID %d'
                msg = msg % (label, pid)
                warnings.warn(msg, BreakpointWarning)
            deferred[label] = (action, oneshot)
            return None
    if self.has_code_breakpoint(pid, address):
        bp = self.get_code_breakpoint(pid, address)
        if bp.get_action() != action:
            bp.set_action(action)
            msg = 'Redefined code breakpoint at %s in process ID %d'
            msg = msg % (label, pid)
            warnings.warn(msg, BreakpointWarning)
    else:
        self.define_code_breakpoint(pid, address, True, action)
        bp = self.get_code_breakpoint(pid, address)
    if oneshot:
        if not bp.is_one_shot():
            self.enable_one_shot_code_breakpoint(pid, address)
    elif not bp.is_enabled():
        self.enable_code_breakpoint(pid, address)
    return bp