from __future__ import annotations
import enum
import itertools
import socket
import sys
from contextlib import contextmanager
from typing import (
import attrs
from outcome import Value
from .. import _core
from ._io_common import wake_all
from ._run import _public
from ._windows_cffi import (
def _refresh_afd(self, base_handle: Handle) -> None:
    waiters = self._afd_waiters[base_handle]
    if waiters.current_op is not None:
        afd_group = waiters.current_op.afd_group
        try:
            _check(kernel32.CancelIoEx(afd_group.handle, waiters.current_op.lpOverlapped))
        except OSError as exc:
            if exc.winerror != ErrorCodes.ERROR_NOT_FOUND:
                raise
        waiters.current_op = None
        afd_group.size -= 1
        self._vacant_afd_groups.add(afd_group)
    flags = 0
    if waiters.read_task is not None:
        flags |= READABLE_FLAGS
    if waiters.write_task is not None:
        flags |= WRITABLE_FLAGS
    if not flags:
        del self._afd_waiters[base_handle]
    else:
        try:
            afd_group = self._vacant_afd_groups.pop()
        except KeyError:
            afd_group = AFDGroup(0, _afd_helper_handle())
            self._register_with_iocp(afd_group.handle, CKeys.AFD_POLL)
            self._all_afd_handles.append(afd_group.handle)
        self._vacant_afd_groups.add(afd_group)
        lpOverlapped = ffi.new('LPOVERLAPPED')
        poll_info: Any = ffi.new('AFD_POLL_INFO *')
        poll_info.Timeout = 2 ** 63 - 1
        poll_info.NumberOfHandles = 1
        poll_info.Exclusive = 0
        poll_info.Handles[0].Handle = base_handle
        poll_info.Handles[0].Status = 0
        poll_info.Handles[0].Events = flags
        try:
            _check(kernel32.DeviceIoControl(afd_group.handle, IoControlCodes.IOCTL_AFD_POLL, poll_info, ffi.sizeof('AFD_POLL_INFO'), poll_info, ffi.sizeof('AFD_POLL_INFO'), ffi.NULL, lpOverlapped))
        except OSError as exc:
            if exc.winerror != ErrorCodes.ERROR_IO_PENDING:
                del self._afd_waiters[base_handle]
                wake_all(waiters, exc)
                return
        op = AFDPollOp(lpOverlapped, poll_info, waiters, afd_group)
        waiters.current_op = op
        self._afd_ops[lpOverlapped] = op
        afd_group.size += 1
        if afd_group.size >= MAX_AFD_GROUP_SIZE:
            self._vacant_afd_groups.remove(afd_group)