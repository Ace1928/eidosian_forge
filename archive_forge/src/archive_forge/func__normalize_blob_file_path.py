from __future__ import absolute_import, division, print_function
import os
import mimetypes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def _normalize_blob_file_path(path, name):
    path_sep = '/'
    if path:
        name = path_sep.join((path, name))
    return path_sep.join(os.path.normpath(name).split(os.path.sep)).strip(path_sep)