import os
import tempfile
from oslo_log import log as logging
from oslo_utils import fileutils
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator import initiator_connector
def _upload_vmdk(self, read_handle, host, port, dc_name, ds_name, cookies, upload_file_path, file_size, cacerts, timeout_secs):
    write_handle = rw_handles.FileWriteHandle(host, port, dc_name, ds_name, cookies, upload_file_path, file_size, cacerts=cacerts)
    image_transfer._start_transfer(read_handle, write_handle, timeout_secs)