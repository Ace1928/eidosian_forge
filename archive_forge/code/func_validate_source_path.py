from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def validate_source_path(self, current):
    """ There can only be one destination, so we use it as the key
            But we want to make sure another relationship is not already using the destination
            It's a bit complicated as the source SVM name can be aliased to a local name if there are conflicts
            So the source can be ansibleSVM: and show locally as ansibleSVM: if there is not conflict or ansibleSVM.1:
            or any alias the user likes.
            And in the input paramters, it may use the remote name or local alias.
        """
    if not current:
        return
    source_path = self.na_helper.safe_get(self.parameters, ['source_endpoint', 'path']) or self.parameters.get('source_path')
    destination_path = self.na_helper.safe_get(self.parameters, ['destination_endpoint', 'path']) or self.parameters.get('destination_path')
    source_cluster = self.na_helper.safe_get(self.parameters, ['source_endpoint', 'cluster']) or self.parameters.get('source_cluster')
    current_source_path = current.pop('source_path', None)
    if source_path and current_source_path and self.parameters.get('validate_source_path'):
        if self.parameters['connection_type'] != 'ontap_ontap':
            if current_source_path != source_path:
                self.module.fail_json(msg='Error: another relationship is present for the same destination with source_path: "%s".  Desired: %s on %s' % (current_source_path, source_path, source_cluster))
            return
        current_source_svm, dummy, dummy = current_source_path.rpartition(':')
        if not current_source_svm:
            self.module.warn('Unexpected source path: %s, skipping validation.' % current_source_path)
        destination_svm, dummy, dummy = destination_path.rpartition(':')
        if not destination_svm:
            self.module.warn('Unexpected destination path: %s, skipping validation.' % destination_path)
        if not current_source_svm or not destination_svm:
            return
        peer_svm, peer_cluster = self.get_svm_peer(current_source_svm, destination_svm)
        if peer_svm is not None:
            real_source_path = current_source_path.replace(current_source_svm, peer_svm, 1)
            if real_source_path != source_path and current_source_path != source_path or (peer_cluster is not None and source_cluster is not None and (source_cluster != peer_cluster)):
                self.module.fail_json(msg='Error: another relationship is present for the same destination with source_path: "%s" (%s on cluster %s).  Desired: %s on %s' % (current_source_path, real_source_path, peer_cluster, source_path, source_cluster))