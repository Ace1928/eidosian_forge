import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def _create_take_action(self, client, app, parsed_args):
    if parsed_args.json:
        blob = osc_utils.read_blob_file_contents(parsed_args.json)
        try:
            template = json.loads(blob)
        except ValueError as e:
            raise exceptions.CommandError('An error occurred when reading template from file %s: %s' % (parsed_args.json, e))
        data = client.node_group_templates.create(**template).to_dict()
    else:
        if not parsed_args.name or not parsed_args.plugin or (not parsed_args.plugin_version) or (not parsed_args.flavor) or (not parsed_args.processes):
            raise exceptions.CommandError('At least --name, --plugin, --plugin-version, --processes, --flavor arguments should be specified or json template should be provided with --json argument')
        configs = None
        if parsed_args.configs:
            blob = osc_utils.read_blob_file_contents(parsed_args.configs)
            try:
                configs = json.loads(blob)
            except ValueError as e:
                raise exceptions.CommandError('An error occurred when reading configs from file %s: %s' % (parsed_args.configs, e))
        shares = None
        if parsed_args.shares:
            blob = osc_utils.read_blob_file_contents(parsed_args.shares)
            try:
                shares = json.loads(blob)
            except ValueError as e:
                raise exceptions.CommandError('An error occurred when reading shares from file %s: %s' % (parsed_args.shares, e))
        compute_client = app.client_manager.compute
        flavor_id = osc_utils.find_resource(compute_client.flavors, parsed_args.flavor).id
        data = create_node_group_templates(client, app, parsed_args, flavor_id, configs, shares)
    return data