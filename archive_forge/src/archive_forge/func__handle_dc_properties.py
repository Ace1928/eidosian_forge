from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _handle_dc_properties(f, connection):
    f.write('dr_import_storages:\n')
    dcs_service = connection.system_service().data_centers_service()
    dcs_list = dcs_service.list()
    clusters = []
    affinity_groups = []
    for dc in dcs_list:
        dc_service = dcs_service.data_center_service(dc.id)
        _write_attached_storage_domains(f, dc_service, dc)
        _add_clusters_and_aff_groups_for_dc(dc_service, clusters, affinity_groups)
    return (clusters, affinity_groups)