from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _get_attrs_network(client_manager, parsed_args):
    attrs = {}
    if parsed_args.name is not None:
        attrs['name'] = parsed_args.name
    if parsed_args.enable:
        attrs['admin_state_up'] = True
    if parsed_args.disable:
        attrs['admin_state_up'] = False
    if parsed_args.share:
        attrs['shared'] = True
    if parsed_args.no_share:
        attrs['shared'] = False
    if parsed_args.enable_port_security:
        attrs['port_security_enabled'] = True
    if parsed_args.disable_port_security:
        attrs['port_security_enabled'] = False
    if 'project' in parsed_args and parsed_args.project is not None:
        identity_client = client_manager.identity
        project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
        attrs['project_id'] = project_id
    if 'availability_zone_hints' in parsed_args and parsed_args.availability_zone_hints is not None:
        attrs['availability_zone_hints'] = parsed_args.availability_zone_hints
    if parsed_args.description is not None:
        attrs['description'] = parsed_args.description
    if parsed_args.mtu:
        attrs['mtu'] = parsed_args.mtu
    if parsed_args.internal:
        attrs['router:external'] = False
    if parsed_args.external:
        attrs['router:external'] = True
    if parsed_args.no_default:
        attrs['is_default'] = False
    if parsed_args.default:
        attrs['is_default'] = True
    if parsed_args.provider_network_type:
        attrs['provider:network_type'] = parsed_args.provider_network_type
    if parsed_args.physical_network:
        attrs['provider:physical_network'] = parsed_args.physical_network
    if parsed_args.segmentation_id:
        attrs['provider:segmentation_id'] = parsed_args.segmentation_id
    if parsed_args.qos_policy is not None:
        network_client = client_manager.network
        _qos_policy = network_client.find_qos_policy(parsed_args.qos_policy, ignore_missing=False)
        attrs['qos_policy_id'] = _qos_policy.id
    if 'no_qos_policy' in parsed_args and parsed_args.no_qos_policy:
        attrs['qos_policy_id'] = None
    if parsed_args.dns_domain is not None:
        attrs['dns_domain'] = parsed_args.dns_domain
    return attrs