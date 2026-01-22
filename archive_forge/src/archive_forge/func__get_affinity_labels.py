from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _get_affinity_labels(connection):
    affinity_labels = []
    affinity_labels_service = connection.system_service().affinity_labels_service()
    affinity_labels_list = affinity_labels_service.list()
    for affinity_label in affinity_labels_list:
        affinity_labels.append(affinity_label.name)
    return affinity_labels