from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _add_clusters_and_aff_groups_for_dc(dc_service, clusters, affinity_groups):
    clusters_service = dc_service.clusters_service()
    attached_clusters_list = clusters_service.list()
    for cluster in attached_clusters_list:
        clusters.append(cluster.name)
        cluster_service = clusters_service.cluster_service(cluster.id)
        _add_affinity_groups_for_cluster(cluster_service, affinity_groups)