import ctypes
import os
import signal
import struct
import threading
from pyglet.libs.x11 import xlib
from pyglet.util import asbytes
def _install_restore_mode_child():
    global _mode_write_pipe
    global _restore_mode_child_installed
    if _restore_mode_child_installed:
        return
    mode_read_pipe, _mode_write_pipe = os.pipe()
    if os.fork() == 0:
        os.close(_mode_write_pipe)
        PR_SET_PDEATHSIG = 1
        libc = ctypes.cdll.LoadLibrary('libc.so.6')
        libc.prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGHUP, 0, 0, 0)

        def _sighup(signum, frame):
            parent_wait_lock.release()
        parent_wait_lock = threading.Lock()
        parent_wait_lock.acquire()
        signal.signal(signal.SIGHUP, _sighup)
        packets = []
        buffer = asbytes('')
        while parent_wait_lock.locked():
            try:
                data = os.read(mode_read_pipe, ModePacket.size)
                buffer += data
                while len(buffer) >= ModePacket.size:
                    packet = ModePacket.decode(buffer[:ModePacket.size])
                    packets.append(packet)
                    buffer = buffer[ModePacket.size:]
            except OSError:
                pass
        for packet in packets:
            packet.set()
        os._exit(0)
    else:
        os.close(mode_read_pipe)
        _restore_mode_child_installed = True