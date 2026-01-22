import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
def _print_cluster(cluster, include_all=False):
    info = cluster._info.copy()
    info['datastore'] = cluster.datastore['type']
    info['datastore_version'] = cluster.datastore['version']
    info['task_name'] = cluster.task['name']
    info['task_description'] = cluster.task['description']
    info.pop('task', None)
    if include_all and hasattr(cluster, 'ip'):
        info['ip'] = ', '.join(cluster.ip)
    instances = info.pop('instances', None)
    if instances:
        info['instance_count'] = len(instances)
    info.pop('links', None)
    utils.print_dict(info)