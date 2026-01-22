import itertools
import json
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class SetBaremetalDeployTemplate(command.Command):
    """Set baremetal deploy template properties."""
    log = logging.getLogger(__name__ + '.SetBaremetalDeployTemplate')

    def get_parser(self, prog_name):
        parser = super(SetBaremetalDeployTemplate, self).get_parser(prog_name)
        parser.add_argument('template', metavar='<template>', help=_('Name or UUID of the deploy template'))
        parser.add_argument('--name', metavar='<name>', help=_('Set unique name of the deploy template. Must be a valid trait name.'))
        parser.add_argument('--steps', metavar='<steps>', help=_DEPLOY_STEPS_HELP)
        parser.add_argument('--extra', metavar='<key=value>', action='append', help=_('Extra to set on this baremetal deploy template (repeat option to set multiple extras).'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        properties = []
        if parsed_args.name:
            name = ['name=%s' % parsed_args.name]
            properties.extend(utils.args_array_to_patch('add', name))
        if parsed_args.steps:
            steps = utils.handle_json_arg(parsed_args.steps, 'deploy steps')
            steps = ['steps=%s' % json.dumps(steps)]
            properties.extend(utils.args_array_to_patch('add', steps))
        if parsed_args.extra:
            properties.extend(utils.args_array_to_patch('add', ['extra/' + x for x in parsed_args.extra]))
        if properties:
            baremetal_client.deploy_template.update(parsed_args.template, properties)
        else:
            self.log.warning('Please specify what to set.')