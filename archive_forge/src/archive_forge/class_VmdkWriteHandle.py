import logging
import ssl
import time
from oslo_utils import excutils
from oslo_utils import netutils
import requests
import urllib.parse as urlparse
from urllib3 import connection as httplib
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware import vim_util
class VmdkWriteHandle(VmdkHandle):
    """VMDK write handle based on HttpNfcLease.

    This class creates a vApp in the specified resource pool and uploads the
    virtual disk contents.
    """

    def __init__(self, session, host, port, rp_ref, vm_folder_ref, import_spec, vmdk_size, http_method='PUT'):
        """Initializes the VMDK write handle with input parameters.

        :param session: valid API session to ESX/VC server
        :param host: ESX/VC server IP address or host name
        :param port: port for connection
        :param rp_ref: resource pool into which the backing VM is imported
        :param vm_folder_ref: VM folder in ESX/VC inventory to use as parent
                              of backing VM
        :param import_spec: import specification of the backing VM
        :param vmdk_size: size of the backing VM's VMDK file
        :param http_method: either PUT or POST
        :raises: VimException, VimFaultException, VimAttributeException,
                 VimSessionOverLoadException, VimConnectionException,
                 ValueError
        """
        self._vmdk_size = vmdk_size
        self._bytes_written = 0
        lease, lease_info = self._create_import_vapp_lease(session, rp_ref, import_spec, vm_folder_ref)
        url, thumbprint = self._find_vmdk_url(lease_info, host, port)
        self._vm_ref = lease_info.entity
        cookies = session.vim.client.cookiejar
        if http_method == 'PUT':
            overwrite = 't'
            content_type = 'binary/octet-stream'
        elif http_method == 'POST':
            overwrite = None
            content_type = 'application/x-vnd.vmware-streamVmdk'
        else:
            raise ValueError('http_method must be either PUT or POST')
        self._conn = self._create_write_connection(http_method, url, vmdk_size, cookies=cookies, overwrite=overwrite, content_type=content_type, ssl_thumbprint=thumbprint)
        super(VmdkWriteHandle, self).__init__(session, lease, url, self._conn)

    def get_imported_vm(self):
        """"Get managed object reference of the VM created for import.

        :raises: VimException
        """
        if self._get_progress() < 100:
            excep_msg = _('Incomplete VMDK upload to %s.') % self._url
            LOG.exception(excep_msg)
            raise exceptions.ImageTransferException(excep_msg)
        return self._vm_ref

    def tell(self):
        return self._bytes_written

    def write(self, data):
        """Write data to the file.

        :param data: data to be written
        :raises: VimConnectionException, VimException
        """
        try:
            self._file_handle.send(data)
            self._bytes_written += len(data)
        except requests.RequestException as excep:
            excep_msg = _('Connection error occurred while writing data to %s.') % self._url
            LOG.exception(excep_msg)
            raise exceptions.VimConnectionException(excep_msg, excep)
        except Exception as excep:
            excep_msg = _('Error occurred while writing data to %s.') % self._url
            LOG.exception(excep_msg)
            raise exceptions.VimException(excep_msg, excep)

    def close(self):
        """Releases the lease and close the connection.

        :raises: VimAttributeException, VimSessionOverLoadException,
                 VimConnectionException
        """
        try:
            self._release_lease()
        except exceptions.ManagedObjectNotFoundException:
            LOG.info('Lease for %(url)s not found.  No need to release.', {'url': self._url})
            return
        except exceptions.VimException:
            LOG.warning('Error occurred while releasing the lease for %s.', self._url, exc_info=True)
        super(VmdkWriteHandle, self).close()
        LOG.debug('Closed VMDK write handle for %s.', self._url)

    def _get_progress(self):
        return float(self._bytes_written) / self._vmdk_size * 100

    def __str__(self):
        return 'VMDK write handle for %s' % self._url