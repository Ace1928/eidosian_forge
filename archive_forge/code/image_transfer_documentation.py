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
Upload the VM's disk file to image service.

    :param context: image service write context
    :param timeout_secs: time in seconds to wait for the upload to complete
    :param image_service: image service handle
    :param image_id: upload destination image ID
    :param kwargs: keyword arguments to configure the source
                   VMDK read handle
    :raises: VimException, VimFaultException, VimAttributeException,
             VimSessionOverLoadException, VimConnectionException,
             ImageTransferException, ValueError
    