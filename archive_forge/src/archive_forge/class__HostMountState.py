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
class _HostMountState(object):
    """A data structure recording all managed mountpoints and the
    attachments in use for each one. _HostMountState ensures that the glance
    node only attempts to mount a single mountpoint in use by multiple
    attachments once, and that it is not unmounted until it is no longer in use
    by any attachments.

    Callers should not create a _HostMountState directly, but should obtain
    it via:

      with mount.get_manager().get_state() as state:
        state.mount(...)

    _HostMountState manages concurrency itself. Independent callers do not need
    to consider interactions between multiple _HostMountState calls when
    designing their own locking.
    """

    class _MountPoint(object):
        """A single mountpoint, and the set of attachments in use on it."""

        def __init__(self):
            self.lock = threading.Lock()
            self.attachments = set()

        def add_attachment(self, vol_name, host):
            self.attachments.add((vol_name, host))

        def remove_attachment(self, vol_name, host):
            self.attachments.remove((vol_name, host))

        def in_use(self):
            return len(self.attachments) > 0

    def __init__(self):
        """Initialise _HostMountState"""
        self.mountpoints = collections.defaultdict(self._MountPoint)

    @contextlib.contextmanager
    def _get_locked(self, mountpoint):
        """Get a locked mountpoint object

        :param mountpoint: The path of the mountpoint whose object we should
                           return.
        :rtype: _HostMountState._MountPoint
        """
        while True:
            mount = self.mountpoints[mountpoint]
            with mount.lock:
                if self.mountpoints[mountpoint] is mount:
                    yield mount
                    break

    def mount(self, fstype, export, vol_name, mountpoint, host, rootwrap_helper, options):
        """Ensure a mountpoint is available for an attachment, mounting it
        if necessary.

        If this is the first attachment on this mountpoint, we will mount it
        with:

          mount -t <fstype> <options> <export> <mountpoint>

        :param fstype: The filesystem type to be passed to mount command.
        :param export: The type-specific identifier of the filesystem to be
                       mounted. e.g. for nfs 'host.example.com:/mountpoint'.
        :param vol_name: The name of the volume on the remote filesystem.
        :param mountpoint: The directory where the filesystem will be
                           mounted on the local compute host.
        :param host: The host the volume will be attached to.
        :param options: An arbitrary list of additional arguments to be
                        passed to the mount command immediate before export
                        and mountpoint.
        """
        LOG.debug('_HostMountState.mount(fstype=%(fstype)s, export=%(export)s, vol_name=%(vol_name)s, %(mountpoint)s, options=%(options)s)', {'fstype': fstype, 'export': export, 'vol_name': vol_name, 'mountpoint': mountpoint, 'options': options})
        with self._get_locked(mountpoint) as mount:
            if not os.path.ismount(mountpoint):
                LOG.debug('Mounting %(mountpoint)s', {'mountpoint': mountpoint})
                os.makedirs(mountpoint)
                mount_cmd = ['mount', '-t', fstype]
                if options is not None:
                    mount_cmd.extend(options)
                mount_cmd.extend([export, mountpoint])
                try:
                    processutils.execute(*mount_cmd, run_as_root=True, root_helper=rootwrap_helper)
                except Exception:
                    if os.path.ismount(mountpoint):
                        LOG.exception(_LE('Error mounting %(fstype)s export %(export)s on %(mountpoint)s. Continuing because mountpount is mounted despite this.'), {'fstype': fstype, 'export': export, 'mountpoint': mountpoint})
                    else:
                        del self.mountpoints[mountpoint]
                        raise
            mount.add_attachment(vol_name, host)
        LOG.debug('_HostMountState.mount() for %(mountpoint)s completed successfully', {'mountpoint': mountpoint})

    def umount(self, vol_name, mountpoint, host, rootwrap_helper):
        """Mark an attachment as no longer in use, and unmount its mountpoint
        if necessary.

        :param vol_name: The name of the volume on the remote filesystem.
        :param mountpoint: The directory where the filesystem is be
                           mounted on the local compute host.
        :param host: The host the volume was attached to.
        """
        LOG.debug('_HostMountState.umount(vol_name=%(vol_name)s, mountpoint=%(mountpoint)s)', {'vol_name': vol_name, 'mountpoint': mountpoint})
        with self._get_locked(mountpoint) as mount:
            try:
                mount.remove_attachment(vol_name, host)
            except KeyError:
                LOG.warning(_LW("Request to remove attachment (%(vol_name)s, %(host)s) from %(mountpoint)s, but we don't think it's in use."), {'vol_name': vol_name, 'host': host, 'mountpoint': mountpoint})
            if not mount.in_use():
                mounted = os.path.ismount(mountpoint)
                if mounted:
                    mounted = self._real_umount(mountpoint, rootwrap_helper)
                if not mounted:
                    del self.mountpoints[mountpoint]
            LOG.debug('_HostMountState.umount() for %(mountpoint)s completed successfully', {'mountpoint': mountpoint})

    def _real_umount(self, mountpoint, rootwrap_helper):
        LOG.debug('Unmounting %(mountpoint)s', {'mountpoint': mountpoint})
        try:
            processutils.execute('umount', mountpoint, run_as_root=True, attempts=3, delay_on_retry=True, root_helper=rootwrap_helper)
        except processutils.ProcessExecutionError as ex:
            LOG.error(_LE("Couldn't unmount %(mountpoint)s: %(reason)s"), {'mountpoint': mountpoint, 'reason': ex})
        if not os.path.ismount(mountpoint):
            try:
                os.rmdir(mountpoint)
            except Exception as ex:
                LOG.error(_LE("Couldn't remove directory %(mountpoint)s: %(reason)s"), {'mountpoint': mountpoint, 'reason': ex})
            return False
        return True