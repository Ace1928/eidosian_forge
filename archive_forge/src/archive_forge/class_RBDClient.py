from __future__ import annotations
import io
from typing import NoReturn, Optional  # noqa: H301
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick import utils
class RBDClient(object):

    def __init__(self, user: str, pool: str, *args, **kwargs):
        self.rbd_user = user
        self.rbd_pool = pool
        self.rados: 'rados.Rados'
        self.rbd: 'rbd.RBD'
        for attr in ['rbd_user', 'rbd_pool']:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, utils.convert_str(val))
        self.rados = kwargs.get('rados', rados)
        self.rbd = kwargs.get('rbd', rbd)
        if self.rados is None:
            raise exception.InvalidParameterValue(err=_('rados module required'))
        if self.rbd is None:
            raise exception.InvalidParameterValue(err=_('rbd module required'))
        self.rbd_conf: str = kwargs.get('conffile', '/etc/ceph/ceph.conf')
        self.rbd_cluster_name: str = kwargs.get('rbd_cluster_name', 'ceph')
        self.client, self.ioctx = self.connect()

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        self.disconnect()

    def connect(self) -> tuple['rados.Rados', 'rados.Ioctx']:
        LOG.debug('opening connection to ceph cluster')
        client = self.rados.Rados(rados_id=self.rbd_user, clustername=self.rbd_cluster_name, conffile=self.rbd_conf)
        try:
            client.connect()
            ioctx = client.open_ioctx(self.rbd_pool)
            return (client, ioctx)
        except self.rados.Error:
            msg = _('Error connecting to ceph cluster.')
            LOG.exception(msg)
            client.shutdown()
            raise exception.BrickException(message=msg)

    def disconnect(self) -> None:
        self.ioctx.close()
        self.client.shutdown()