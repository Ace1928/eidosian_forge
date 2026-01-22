import os
import tempfile
from oslo_log import log as logging
from oslo_utils import fileutils
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator import initiator_connector
def _download_vmdk(self, tmp_file_path, session, backing, vmdk_path, vmdk_size):
    with open(tmp_file_path, 'wb') as tmp_file:
        image_transfer.copy_stream_optimized_disk(None, self._timeout, tmp_file, session=session, host=self._ip, port=self._port, vm=backing, vmdk_file_path=vmdk_path, vmdk_size=vmdk_size)