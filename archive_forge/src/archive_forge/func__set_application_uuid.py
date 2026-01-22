from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def _set_application_uuid(self):
    """Use REST application/applications to get application uuid"""
    api = 'application/applications'
    query = {'svm.name': self.svm_name, 'name': self.app_name}
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error is None and record is not None:
        self.app_uuid = record['uuid']
    return (None, error)