import asyncio
import os
import signal
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from ..connect import KernelConnectionInfo, LocalPortCache
from ..launcher import launch_kernel
from ..localinterfaces import is_local_ip, local_ips
from .provisioner_base import KernelProvisionerBase
@staticmethod
def _tolerate_no_process(os_error: OSError) -> None:
    if sys.platform == 'win32':
        if os_error.winerror != 5:
            raise
    else:
        from errno import ESRCH
        if not isinstance(os_error, ProcessLookupError) or os_error.errno != ESRCH:
            raise