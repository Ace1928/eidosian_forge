import os
import numpy as np
import threading
from time import time
from .. import config, logging
class ResourceMonitor(threading.Thread):
    """
    A ``Thread`` to monitor a specific PID with a certain frequence
    to a file
    """

    def __init__(self, pid, freq=5, fname=None, python=True):
        import psutil
        if freq < 0.2:
            raise RuntimeError('Frequency (%0.2fs) cannot be lower than 0.2s' % freq)
        if fname is None:
            fname = '.proc-%d_time-%s_freq-%0.2f' % (pid, time(), freq)
        self._fname = os.path.abspath(fname)
        self._logfile = open(self._fname, 'w')
        self._freq = freq
        self._python = python
        self._process = psutil.Process(pid)
        self._sample(cpu_interval=0.2)
        threading.Thread.__init__(self)
        self._event = threading.Event()

    @property
    def fname(self):
        """Get/set the internal filename"""
        return self._fname

    def stop(self):
        """Stop monitoring."""
        if not self._event.is_set():
            self._event.set()
            self.join()
            self._sample()
            self._logfile.flush()
            self._logfile.close()
        retval = {'mem_peak_gb': None, 'cpu_percent': None}
        vals = np.loadtxt(self._fname, delimiter=',')
        if vals.size:
            vals = np.atleast_2d(vals)
            retval['mem_peak_gb'] = vals[:, 2].max() / 1024
            retval['cpu_percent'] = vals[:, 1].max()
            retval['prof_dict'] = {'time': vals[:, 0].tolist(), 'cpus': vals[:, 1].tolist(), 'rss_GiB': (vals[:, 2] / 1024).tolist(), 'vms_GiB': (vals[:, 3] / 1024).tolist()}
        return retval

    def _sample(self, cpu_interval=None):
        cpu = 0.0
        rss = 0.0
        vms = 0.0
        try:
            with self._process.oneshot():
                cpu += self._process.cpu_percent(interval=cpu_interval)
                mem_info = self._process.memory_info()
                rss += mem_info.rss
                vms += mem_info.vms
        except psutil.NoSuchProcess:
            pass
        try:
            children = self._process.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in children:
            try:
                with child.oneshot():
                    cpu += child.cpu_percent()
                    mem_info = child.memory_info()
                    rss += mem_info.rss
                    vms += mem_info.vms
            except psutil.NoSuchProcess:
                pass
        print('%f,%f,%f,%f' % (time(), cpu, rss / _MB, vms / _MB), file=self._logfile)
        self._logfile.flush()

    def run(self):
        """Core monitoring function, called by start()"""
        start_time = time()
        wait_til = start_time
        while not self._event.is_set():
            self._sample()
            wait_til += self._freq
            self._event.wait(max(0, wait_til - time()))