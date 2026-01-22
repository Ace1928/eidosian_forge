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
@staticmethod
def _create_export_vm_lease(session, vm_ref):
    """Create and wait for HttpNfcLease lease for VM export."""
    LOG.debug('Creating HttpNfcLease lease for exporting VM: %s.', vm_ref)
    lease = session.invoke_api(session.vim, 'ExportVm', vm_ref)
    LOG.debug('Lease: %(lease)s obtained for exporting VM: %(vm_ref)s.', {'lease': lease, 'vm_ref': vm_ref})
    session.wait_for_lease_ready(lease)
    LOG.debug('Invoking VIM API for reading info of lease: %s.', lease)
    lease_info = session.invoke_api(vim_util, 'get_object_property', session.vim, lease, 'info')
    return (lease, lease_info)