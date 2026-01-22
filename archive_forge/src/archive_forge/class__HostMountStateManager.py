import collections
import contextlib
import logging
import os
import socket
import threading
from oslo_concurrency import processutils
from oslo_config import cfg
from glance_store import exceptions
from glance_store.i18n import _LE, _LW
class _HostMountStateManager(metaclass=HostMountStateManagerMeta):
    """A global manager of filesystem mounts.

    _HostMountStateManager manages a _HostMountState object for the current
    glance node. Primarily it creates one on object initialization and returns
    it via get_state().

    _HostMountStateManager manages concurrency itself. Independent callers do
    not need to consider interactions between multiple _HostMountStateManager
    calls when designing their own locking.

    """
    state = None
    use_count = 0
    cond = threading.Condition()

    def __init__(self, host):
        """Initialise a new _HostMountState

        We will block before creating a new state until all operations
        using a previous state have completed.

        :param host: host
        """
        self.host = host
        while self.use_count != 0:
            self.cond.wait()
        if self.state is None:
            LOG.debug('Initialising _HostMountState')
            self.state = _HostMountState()
            backends = []
            enabled_backends = CONF.enabled_backends
            if enabled_backends:
                for backend in enabled_backends:
                    if enabled_backends[backend] == 'cinder':
                        backends.append(backend)
            else:
                backends.append('glance_store')
            for backend in backends:
                mountpoint = getattr(CONF, backend).cinder_mount_point_base
                mountpoint = os.path.join(mountpoint, 'nfs')
                rootwrap = getattr(CONF, backend).rootwrap_config
                rootwrap = 'sudo glance-rootwrap %s' % rootwrap
                dirs = []
                if os.path.isdir(mountpoint):
                    dirs = os.listdir(mountpoint)
                else:
                    continue
                if not dirs:
                    return
                for dir in dirs:
                    dir = os.path.join(mountpoint, dir)
                    with self.get_state() as mount_state:
                        if os.path.exists(dir) and (not os.path.ismount(dir)):
                            try:
                                os.rmdir(dir)
                            except Exception as ex:
                                LOG.debug("Couldn't remove directory %(mountpoint)s: %(reason)s", {'mountpoint': mountpoint, 'reason': ex})
                        else:
                            mount_state.umount(None, dir, HOST, rootwrap)

    @contextlib.contextmanager
    def get_state(self):
        """Return the current mount state.

        _HostMountStateManager will not permit a new state object to be
        created while any previous state object is still in use.

        :rtype: _HostMountState
        """
        with self.cond:
            state = self.state
            if state is None:
                LOG.error('Host not initialized')
                raise exceptions.HostNotInitialized(host=self.host)
            self.use_count += 1
        try:
            LOG.debug('Got _HostMountState')
            yield state
        finally:
            with self.cond:
                self.use_count -= 1
                self.cond.notify_all()