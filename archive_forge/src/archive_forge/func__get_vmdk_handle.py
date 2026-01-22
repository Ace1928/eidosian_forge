import logging
import tarfile
from eventlet import timeout
from oslo_utils import units
from oslo_vmware._i18n import _
from oslo_vmware.common import loopingcall
from oslo_vmware import constants
from oslo_vmware import exceptions
from oslo_vmware import image_util
from oslo_vmware.objects import datastore as ds_obj
from oslo_vmware import rw_handles
from oslo_vmware import vim_util
def _get_vmdk_handle(ova_handle):
    with tarfile.open(mode='r|', fileobj=ova_handle) as tar:
        vmdk_name = None
        for tar_info in tar:
            if tar_info and tar_info.name.endswith('.ovf'):
                vmdk_name = image_util.get_vmdk_name_from_ovf(tar.extractfile(tar_info))
            elif vmdk_name and tar_info.name.startswith(vmdk_name):
                return tar.extractfile(tar_info)