import abc
import contextlib
import logging
import openstack.exceptions
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from openstackclient.i18n import _
from openstackclient.network import utils
class NeutronCommandWithExtraArgs(command.Command):
    """Create and Update commands with additional extra properties.

    Extra properties can be passed to the command and are then send to the
    Neutron as given to the command.
    """
    _allowed_types_dict = {'bool': utils.str2bool, 'dict': utils.str2dict, 'list': utils.str2list, 'int': int, 'str': str}

    def _get_property_converter(self, _property):
        if 'type' not in _property:
            converter = str
        else:
            converter = self._allowed_types_dict.get(_property['type'])
        if not converter:
            raise exceptions.CommandError(_('Type {property_type} of property {name} is not supported').format(property_type=_property['type'], name=_property['name']))
        return converter

    def _parse_extra_properties(self, extra_properties):
        result = {}
        if extra_properties:
            for _property in extra_properties:
                converter = self._get_property_converter(_property)
                result[_property['name']] = converter(_property['value'])
        return result

    def get_parser(self, prog_name):
        parser = super(NeutronCommandWithExtraArgs, self).get_parser(prog_name)
        parser.add_argument('--extra-property', metavar='type=<property_type>,name=<property_name>,value=<property_value>', dest='extra_properties', action=parseractions.MultiKeyValueAction, required_keys=['name', 'value'], optional_keys=['type'], help=_("Additional parameters can be passed using this property. Default type of the extra property is string ('str'), but other types can be used as well. Available types are: 'dict', 'list', 'str', 'bool', 'int'. In case of 'list' type, 'value' can be semicolon-separated list of values. For 'dict' value is semicolon-separated list of the key:value pairs."))
        return parser