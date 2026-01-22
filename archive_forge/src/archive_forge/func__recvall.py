import os
import socket
from subprocess import Popen, PIPE
from contextlib import contextmanager
import numpy as np
from ase.calculators.calculator import (Calculator, all_changes,
import ase.units as units
from ase.utils import IOContext
from ase.stress import full_3x3_to_voigt_6_stress
def _recvall(self, nbytes):
    """Repeatedly read chunks until we have nbytes.

        Normally we get all bytes in one read, but that is not guaranteed."""
    remaining = nbytes
    chunks = []
    while remaining > 0:
        chunk = self.socket.recv(remaining)
        if len(chunk) == 0:
            raise SocketClosed()
        chunks.append(chunk)
        remaining -= len(chunk)
    msg = b''.join(chunks)
    assert len(msg) == nbytes and remaining == 0
    return msg