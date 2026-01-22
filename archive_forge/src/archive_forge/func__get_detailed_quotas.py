import argparse
import itertools
import logging
import sys
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
def _get_detailed_quotas(self, parsed_args):
    project_info = get_project(self.app, parsed_args.project)
    project = project_info['id']
    quotas = {}
    if parsed_args.compute:
        quotas.update(get_compute_quotas(self.app, project, detail=parsed_args.detail))
    if parsed_args.network:
        quotas.update(get_network_quotas(self.app, project, detail=parsed_args.detail))
    if parsed_args.volume:
        quotas.update(get_volume_quotas(self.app, parsed_args, detail=parsed_args.detail))
    result = []
    for resource, values in quotas.items():
        if isinstance(values, dict):
            result.append({'resource': resource, 'in_use': values.get('in_use'), 'reserved': values.get('reserved'), 'limit': values.get('limit')})
    columns = ('resource', 'in_use', 'reserved', 'limit')
    column_headers = ('Resource', 'In Use', 'Reserved', 'Limit')
    return (column_headers, (utils.get_dict_properties(s, columns) for s in result))