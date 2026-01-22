from __future__ import annotations
from collections import namedtuple
from itertools import starmap
from multiprocessing import Pipe, Process, current_process
from time import sleep
from timeit import default_timer
from dask.callbacks import Callback
from dask.utils import import_required
class _Tracker(Process):
    """Background process for tracking resource usage"""

    def __init__(self, dt=1):
        super().__init__()
        self.daemon = True
        self.dt = dt
        self.parent_pid = current_process().pid
        self.parent_conn, self.child_conn = Pipe()

    def shutdown(self):
        if not self.parent_conn.closed:
            self.parent_conn.send('shutdown')
            self.parent_conn.close()
        self.join()

    def _update_pids(self, pid):
        return [self.parent] + [p for p in self.parent.children() if p.pid != pid and p.status() != 'zombie']

    def run(self):
        psutil = import_required('psutil', 'Tracking resource usage requires `psutil` to be installed')
        self.parent = psutil.Process(self.parent_pid)
        pid = current_process()
        data = []
        while True:
            try:
                msg = self.child_conn.recv()
            except KeyboardInterrupt:
                continue
            if msg == 'shutdown':
                break
            elif msg == 'collect':
                ps = self._update_pids(pid)
                while not data or not self.child_conn.poll():
                    tic = default_timer()
                    mem = cpu = 0
                    for p in ps:
                        try:
                            mem2 = p.memory_info().rss
                            cpu2 = p.cpu_percent()
                        except Exception:
                            pass
                        else:
                            mem += mem2
                            cpu += cpu2
                    data.append((tic, mem / 1000000.0, cpu))
                    sleep(self.dt)
            elif msg == 'send_data':
                self.child_conn.send(data)
                data = []
        self.child_conn.close()