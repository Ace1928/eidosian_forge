import contextlib
import ctypes
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
class ClusterContextManager(object):
    _CLUSTER_HANDLE = 0
    _NODE_HANDLE = 1
    _GROUP_HANDLE = 2
    _RESOURCE_HANDLE = 3
    _ENUM_HANDLE = 4
    _HANDLE_TYPES = [_CLUSTER_HANDLE, _NODE_HANDLE, _GROUP_HANDLE, _RESOURCE_HANDLE, _ENUM_HANDLE]

    def __init__(self):
        self._clusapi_utils = ClusApiUtils()

    def open_cluster(self, cluster_name=None):
        return self._open(cluster_name, self._CLUSTER_HANDLE)

    def open_cluster_group(self, group_name, cluster_handle=None):
        return self._open(group_name, self._GROUP_HANDLE, cluster_handle)

    def open_cluster_resource(self, resource_name, cluster_handle=None):
        return self._open(resource_name, self._RESOURCE_HANDLE, cluster_handle)

    def open_cluster_node(self, node_name, cluster_handle=None):
        return self._open(node_name, self._NODE_HANDLE, cluster_handle)

    def open_cluster_enum(self, object_type, cluster_handle=None):
        return self._open(object_type, self._ENUM_HANDLE, cluster_handle)

    def _check_handle_type(self, handle_type):
        if handle_type not in self._HANDLE_TYPES:
            err_msg = _('Invalid cluster handle type: %(handle_type)s. Allowed handle types: %(allowed_types)s.')
            raise exceptions.Invalid(err_msg % dict(handle_type=handle_type, allowed_types=self._HANDLE_TYPES))

    def _close(self, handle, handle_type):
        self._check_handle_type(handle_type)
        if not handle:
            return
        cutils = self._clusapi_utils
        helper_map = {self._CLUSTER_HANDLE: cutils.close_cluster, self._RESOURCE_HANDLE: cutils.close_cluster_resource, self._GROUP_HANDLE: cutils.close_cluster_group, self._NODE_HANDLE: cutils.close_cluster_node, self._ENUM_HANDLE: cutils.close_cluster_enum}
        helper_map[handle_type](handle)

    @contextlib.contextmanager
    def _open(self, name=None, handle_type=_CLUSTER_HANDLE, cluster_handle=None):
        self._check_handle_type(handle_type)
        ext_cluster_handle = cluster_handle is not None
        handle = None
        try:
            if not cluster_handle:
                cluster_name = name if handle_type == self._CLUSTER_HANDLE else None
                cluster_handle = self._clusapi_utils.open_cluster(cluster_name)
            cutils = self._clusapi_utils
            helper_map = {self._CLUSTER_HANDLE: lambda x, y: x, self._RESOURCE_HANDLE: cutils.open_cluster_resource, self._GROUP_HANDLE: cutils.open_cluster_group, self._NODE_HANDLE: cutils.open_cluster_node, self._ENUM_HANDLE: cutils.open_cluster_enum}
            handle = helper_map[handle_type](cluster_handle, name)
            yield handle
        except exceptions.ClusterWin32Exception as win32_ex:
            if win32_ex.error_code in w_const.CLUSTER_NOT_FOUND_ERROR_CODES:
                err_msg = _('Could not find the specified cluster object. Object type: %(obj_type)s. Object name: %(name)s.')
                raise exceptions.ClusterObjectNotFound(err_msg % dict(obj_type=handle_type, name=name))
            else:
                raise
        finally:
            if handle_type != self._CLUSTER_HANDLE:
                self._close(handle, handle_type)
            if not ext_cluster_handle:
                self._close(cluster_handle, self._CLUSTER_HANDLE)