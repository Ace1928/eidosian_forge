import abc
import contextlib
import logging
import openstack.exceptions
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from openstackclient.i18n import _
from openstackclient.network import utils
def _get_property_converter(self, _property):
    if 'type' not in _property:
        converter = str
    else:
        converter = self._allowed_types_dict.get(_property['type'])
    if not converter:
        raise exceptions.CommandError(_('Type {property_type} of property {name} is not supported').format(property_type=_property['type'], name=_property['name']))
    return converter