from __future__ import annotations
import atexit
import time
import weakref
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast
from pymongo import common, periodic_executor
from pymongo._csot import MovingMinimum
from pymongo.errors import NotPrimaryError, OperationFailure, _OperationCancelled
from pymongo.hello import Hello
from pymongo.lock import _create_lock
from pymongo.periodic_executor import _shutdown_executors
from pymongo.pool import _is_faas
from pymongo.read_preferences import MovingAverage
from pymongo.server_description import ServerDescription
from pymongo.srv_resolver import _SrvResolver
class _RttMonitor(MonitorBase):

    def __init__(self, topology: Topology, topology_settings: TopologySettings, pool: Pool):
        """Maintain round trip times for a server.

        The Topology is weakly referenced.
        """
        super().__init__(topology, 'pymongo_server_rtt_thread', topology_settings.heartbeat_frequency, common.MIN_HEARTBEAT_INTERVAL)
        self._pool = pool
        self._moving_average = MovingAverage()
        self._moving_min = MovingMinimum()
        self._lock = _create_lock()

    def close(self) -> None:
        self.gc_safe_close()
        self._pool.reset()

    def add_sample(self, sample: float) -> None:
        """Add a RTT sample."""
        with self._lock:
            self._moving_average.add_sample(sample)
            self._moving_min.add_sample(sample)

    def get(self) -> tuple[Optional[float], float]:
        """Get the calculated average, or None if no samples yet and the min."""
        with self._lock:
            return (self._moving_average.get(), self._moving_min.get())

    def reset(self) -> None:
        """Reset the average RTT."""
        with self._lock:
            self._moving_average.reset()
            self._moving_min.reset()

    def _run(self) -> None:
        try:
            rtt = self._ping()
            self.add_sample(rtt)
        except ReferenceError:
            self.close()
        except Exception:
            self._pool.reset()

    def _ping(self) -> float:
        """Run a "hello" command and return the RTT."""
        with self._pool.checkout() as conn:
            if self._executor._stopped:
                raise Exception('_RttMonitor closed')
            start = time.monotonic()
            conn.hello()
            return time.monotonic() - start