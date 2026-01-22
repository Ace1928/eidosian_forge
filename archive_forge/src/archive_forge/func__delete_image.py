import contextlib
import logging
import math
import urllib
from eventlet import tpool
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import eventletutils
from oslo_utils import units
from glance_store import capabilities
from glance_store.common import utils
from glance_store import driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LI, _LW
from glance_store import location
def _delete_image(self, target_pool, image_name, snapshot_name=None, context=None):
    """
        Delete RBD image and snapshot.

        :param image_name: Image's name
        :param snapshot_name: Image snapshot's name

        :raises: NotFound if image does not exist;
                InUseByStore if image is in use or snapshot unprotect failed
        """
    with self.get_connection(conffile=self.conf_file, rados_id=self.user) as conn:
        with conn.open_ioctx(target_pool) as ioctx:
            try:
                if snapshot_name is not None:
                    with rbd.Image(ioctx, image_name) as image:
                        try:
                            self._unprotect_snapshot(image, snapshot_name)
                            image.remove_snap(snapshot_name)
                        except rbd.ImageNotFound as exc:
                            msg = _('Snap Operating Exception %(snap_exc)s Snapshot does not exist.') % {'snap_exc': exc}
                            LOG.debug(msg)
                        except rbd.ImageBusy as exc:
                            log_msg = _LW('Snap Operating Exception %(snap_exc)s Snapshot is in use.') % {'snap_exc': exc}
                            LOG.warning(log_msg)
                            raise exceptions.InUseByStore()
                self.RBDProxy().remove(ioctx, image_name)
            except rbd.ImageHasSnapshots:
                log_msg = _LW('Unable to remove image %(img_name)s: it has snapshot(s) left; trashing instead') % {'img_name': image_name}
                LOG.warning(log_msg)
                with rbd.Image(ioctx, image_name) as image:
                    try:
                        rbd.RBD().trash_move(ioctx, image_name)
                        LOG.debug('Moved %s to trash', image_name)
                    except rbd.ImageBusy:
                        LOG.warning(_('Unable to move in-use image to trash'))
                        raise exceptions.InUseByStore()
                    return
                raise exceptions.HasSnapshot()
            except rbd.ImageBusy:
                log_msg = _LW('Remove image %(img_name)s failed. It is in use.') % {'img_name': image_name}
                LOG.warning(log_msg)
                raise exceptions.InUseByStore()
            except rbd.ImageNotFound:
                msg = _('RBD image %s does not exist') % image_name
                raise exceptions.NotFound(message=msg)