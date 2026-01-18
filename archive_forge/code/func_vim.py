import logging
from oslo_concurrency import lockutils
from oslo_context import context
from oslo_utils import excutils
from oslo_utils import reflection
from oslo_vmware._i18n import _
from oslo_vmware.common import loopingcall
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware import vim
from oslo_vmware import vim_util
@property
def vim(self):
    if not self._vim:
        self._vim = vim.Vim(protocol=self._scheme, host=self._host, port=self._port, wsdl_url=self._vim_wsdl_loc, cacert=self._cacert, insecure=self._insecure, pool_maxsize=self._pool_size, connection_timeout=self._connection_timeout, op_id_prefix=self._op_id_prefix)
    return self._vim