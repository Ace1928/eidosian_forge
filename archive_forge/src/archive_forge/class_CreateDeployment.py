import logging
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib import utils
from oslo_serialization import jsonutils
from heatclient._i18n import _
from heatclient.common import deployment_utils
from heatclient.common import format_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
class CreateDeployment(format_utils.YamlFormat):
    """Create a software deployment."""
    log = logging.getLogger(__name__ + '.CreateDeployment')

    def get_parser(self, prog_name):
        parser = super(CreateDeployment, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<deployment-name>', help=_('Name of the derived config associated with this deployment. This is used to apply a sort order to the list of configurations currently deployed to the server.'))
        parser.add_argument('--input-value', metavar='<key=value>', action='append', help=_('Input value to set on the deployment. This can be specified multiple times.'))
        parser.add_argument('--action', metavar='<action>', default='UPDATE', help=_('Name of an action for this deployment. This can be a custom action, or one of CREATE, UPDATE, DELETE, SUSPEND, RESUME. Default is UPDATE'))
        parser.add_argument('--config', metavar='<config>', help=_('ID of the configuration to deploy'))
        parser.add_argument('--signal-transport', metavar='<signal-transport>', default='TEMP_URL_SIGNAL', help=_('How the server should signal to heat with the deployment output values. TEMP_URL_SIGNAL will create a Swift TempURL to be signaled via HTTP PUT. ZAQAR_SIGNAL will create a dedicated zaqar queue to be signaled using the provided keystone credentials.NO_SIGNAL will result in the resource going to the COMPLETE state without waiting for any signal'))
        parser.add_argument('--container', metavar='<container>', help=_('Optional name of container to store TEMP_URL_SIGNAL objects in. If not specified a container will be created with a name derived from the DEPLOY_NAME'))
        parser.add_argument('--timeout', metavar='<timeout>', type=int, default=60, help=_('Deployment timeout in minutes'))
        parser.add_argument('--server', metavar='<server>', required=True, help=_('ID of the server being deployed to'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.orchestration
        config = {}
        if parsed_args.config:
            try:
                config = client.software_configs.get(parsed_args.config)
            except heat_exc.HTTPNotFound:
                msg = _('Software configuration not found: %s') % parsed_args.config
                raise exc.CommandError(msg)
        derived_params = deployment_utils.build_derived_config_params(parsed_args.action, config, parsed_args.name, heat_utils.format_parameters(parsed_args.input_value, False), parsed_args.server, parsed_args.signal_transport, signal_id=deployment_utils.build_signal_id(client, parsed_args))
        derived_config = client.software_configs.create(**derived_params)
        sd = client.software_deployments.create(config_id=derived_config.id, server_id=parsed_args.server, action=parsed_args.action, status='IN_PROGRESS')
        return zip(*sorted(sd.to_dict().items()))