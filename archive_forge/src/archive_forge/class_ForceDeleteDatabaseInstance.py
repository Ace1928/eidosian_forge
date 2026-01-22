import argparse
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient.osc.v1 import base
from troveclient import utils as trove_utils
class ForceDeleteDatabaseInstance(command.Command):
    _description = _('Force delete an instance.')

    def get_parser(self, prog_name):
        parser = super(ForceDeleteDatabaseInstance, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('ID or name of the instance'))
        return parser

    def take_action(self, parsed_args):
        db_instances = self.app.client_manager.database.instances
        instance = osc_utils.find_resource(db_instances, parsed_args.instance)
        db_instances.reset_status(instance)
        try:
            db_instances.delete(instance)
        except Exception as e:
            msg = _('Failed to delete instance %(instance)s: %(e)s') % {'instance': parsed_args.instance, 'e': e}
            raise exceptions.CommandError(msg)