from __future__ import annotations
import os
import sys
from typing import Iterable, List, Optional, Tuple, cast
from twisted.internet.defer import Deferred
from twisted.internet.error import ProcessTerminated
from twisted.internet.protocol import ProcessProtocol
from twisted.python.failure import Failure
def _process_finished(self, pp: TestProcessProtocol, cmd: List[str], check_code: bool) -> Tuple[int, bytes, bytes]:
    if pp.exitcode and check_code:
        msg = f'process {cmd} exit with code {pp.exitcode}'
        msg += f'\n>>> stdout <<<\n{pp.out.decode()}'
        msg += '\n'
        msg += f'\n>>> stderr <<<\n{pp.err.decode()}'
        raise RuntimeError(msg)
    return (cast(int, pp.exitcode), pp.out, pp.err)