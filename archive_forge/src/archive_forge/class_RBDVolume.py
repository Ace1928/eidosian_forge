from __future__ import annotations
import io
from typing import NoReturn, Optional  # noqa: H301
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick import utils
class RBDVolume(object):
    """Context manager for dealing with an existing rbd volume."""

    def __init__(self, client: RBDClient, name: str, snapshot: Optional[str]=None, read_only: bool=False):
        if snapshot is not None:
            snapshot = utils.convert_str(snapshot)
        try:
            self.image = client.rbd.Image(client.ioctx, utils.convert_str(name), snapshot=snapshot, read_only=read_only)
        except client.rbd.Error:
            LOG.exception('error opening rbd image %s', name)
            client.disconnect()
            raise
        self.name = name
        self.client = client

    def close(self) -> None:
        try:
            self.image.close()
        finally:
            self.client.disconnect()

    def __enter__(self) -> 'RBDVolume':
        return self

    def __exit__(self, type_, value, traceback) -> None:
        self.close()

    def __getattr__(self, attrib):
        return getattr(self.image, attrib)