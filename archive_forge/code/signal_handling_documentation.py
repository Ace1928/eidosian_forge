import signal
import threading
from . import IS_WINDOWS
from torch._C import _set_worker_pids, _remove_worker_pids  # noqa: F401
from torch._C import _error_if_any_worker_fails, _set_worker_signal_handlers  # noqa: F401
Whether SIGCHLD handler is set for DataLoader worker failures. Only one
handler needs to be set for all DataLoaders in a process.