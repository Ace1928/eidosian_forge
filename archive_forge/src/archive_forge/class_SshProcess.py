from __future__ import annotations
import dataclasses
import itertools
import json
import os
import random
import re
import subprocess
import shlex
import typing as t
from .encoding import (
from .util import (
from .config import (
class SshProcess:
    """Wrapper around an SSH process."""

    def __init__(self, process: t.Optional[subprocess.Popen]) -> None:
        self._process = process
        self.pending_forwards: t.Optional[list[tuple[str, int]]] = None
        self.forwards: dict[tuple[str, int], int] = {}

    def terminate(self) -> None:
        """Terminate the SSH process."""
        if not self._process:
            return
        try:
            self._process.terminate()
        except Exception:
            pass

    def wait(self) -> None:
        """Wait for the SSH process to terminate."""
        if not self._process:
            return
        self._process.wait()

    def collect_port_forwards(self) -> dict[tuple[str, int], int]:
        """Collect port assignments for dynamic SSH port forwards."""
        errors: list[str] = []
        display.info('Collecting %d SSH port forward(s).' % len(self.pending_forwards), verbosity=2)
        while self.pending_forwards:
            if self._process:
                line_bytes = self._process.stderr.readline()
                if not line_bytes:
                    if errors:
                        details = ':\n%s' % '\n'.join(errors)
                    else:
                        details = '.'
                    raise ApplicationError('SSH port forwarding failed%s' % details)
                line = to_text(line_bytes).strip()
                match = re.search('^Allocated port (?P<src_port>[0-9]+) for remote forward to (?P<dst_host>[^:]+):(?P<dst_port>[0-9]+)$', line)
                if not match:
                    if re.search('^Warning: Permanently added .* to the list of known hosts\\.$', line):
                        continue
                    display.warning('Unexpected SSH port forwarding output: %s' % line, verbosity=2)
                    errors.append(line)
                    continue
                src_port = int(match.group('src_port'))
                dst_host = str(match.group('dst_host'))
                dst_port = int(match.group('dst_port'))
                dst = (dst_host, dst_port)
            else:
                dst = self.pending_forwards[0]
                src_port = random.randint(40000, 50000)
            self.pending_forwards.remove(dst)
            self.forwards[dst] = src_port
        display.info('Collected %d SSH port forward(s):\n%s' % (len(self.forwards), '\n'.join(('%s -> %s:%s' % (src_port, dst[0], dst[1]) for dst, src_port in sorted(self.forwards.items())))), verbosity=2)
        return self.forwards