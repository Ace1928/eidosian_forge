from __future__ import annotations
import io
from typing import NoReturn, Optional  # noqa: H301
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick import utils
class RBDVolumeIOWrapper(io.RawIOBase):
    """Enables LibRBD.Image objects to be treated as Python IO objects.

    Calling unimplemented interfaces will raise IOError.
    """

    def __init__(self, rbd_volume: RBDImageMetadata):
        super(RBDVolumeIOWrapper, self).__init__()
        self._rbd_volume = rbd_volume
        self._offset = 0

    def _inc_offset(self, length: int) -> None:
        self._offset += length

    @property
    def rbd_image(self) -> 'rbd.Image':
        return self._rbd_volume.image

    @property
    def rbd_user(self) -> str:
        return self._rbd_volume.user

    @property
    def rbd_pool(self) -> str:
        return self._rbd_volume.pool

    @property
    def rbd_conf(self) -> str:
        return self._rbd_volume.conf

    def read(self, length: Optional[int]=None) -> bytes:
        offset = self._offset
        total = int(self._rbd_volume.image.size())
        if offset >= total:
            return b''
        if length is None:
            length = total
        if offset + length > total:
            length = total - offset
        try:
            data = self._rbd_volume.image.read(int(offset), int(length))
        except Exception:
            LOG.exception('Exception encountered during image read')
            raise
        self._inc_offset(length)
        return data

    def write(self, data) -> None:
        self._rbd_volume.image.write(data, self._offset)
        self._inc_offset(len(data))

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int=0):
        if whence == 0:
            new_offset = offset
        elif whence == 1:
            new_offset = self._offset + offset
        elif whence == 2:
            new_offset = self._rbd_volume.image.size()
            new_offset += offset
        else:
            raise IOError(_('Invalid argument - whence=%s not supported') % whence)
        if new_offset < 0:
            raise IOError(_('Invalid argument'))
        self._offset = new_offset

    def tell(self) -> int:
        return self._offset

    def flush(self) -> None:
        super().flush()
        try:
            self.rbd_image.require_not_closed()
        except rbd.InvalidArgument:
            LOG.warning("RBDVolumeIOWrapper's underlying image %s was closed directly (probably by the GC) instead of through the wrapper", self.rbd_image.name)
            return
        try:
            self.rbd_image.flush()
        except AttributeError:
            LOG.warning('flush() not supported in this version of librbd')

    def fileno(self) -> NoReturn:
        """RBD does not have support for fileno() so we raise IOError.

        Raising IOError is recommended way to notify caller that interface is
        not supported - see http://docs.python.org/2/library/io.html#io.IOBase
        """
        raise IOError(_('fileno() not supported by RBD()'))

    def close(self) -> None:
        if not self.closed:
            super().close()
            self.rbd_image.close()