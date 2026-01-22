import abc
import contextlib
import logging
import openstack.exceptions
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from openstackclient.i18n import _
from openstackclient.network import utils
class NetDetectionMixin(metaclass=abc.ABCMeta):
    """Convenience methods for nova-network vs. neutron decisions.

    A live environment detects which network type it is running and creates its
    parser with only the options relevant to that network type.

    But the command classes are used for docs builds as well, and docs must
    present the options for both network types, often qualified accordingly.
    """

    @property
    def _network_type(self):
        """Discover whether the running cloud is using neutron or nova-network.

        :return:
            * ``NET_TYPE_NEUTRON`` if neutron is detected
            * ``NET_TYPE_COMPUTE`` if running in a cloud but neutron is not
              detected.
            * ``None`` if not running in a cloud, which hopefully means we're
              building docs.
        """
        if not hasattr(self, '_net_type'):
            try:
                if self.app.client_manager.is_network_endpoint_enabled():
                    net_type = _NET_TYPE_NEUTRON
                else:
                    net_type = _NET_TYPE_COMPUTE
            except AttributeError:
                LOG.warning('%s: Could not detect a network type. Assuming we are building docs.', self.__class__.__name__)
                net_type = None
            self._net_type = net_type
        return self._net_type

    @property
    def is_neutron(self):
        return self._network_type is _NET_TYPE_NEUTRON

    @property
    def is_nova_network(self):
        return self._network_type is _NET_TYPE_COMPUTE

    @property
    def is_docs_build(self):
        return self._network_type is None

    def enhance_help_neutron(self, _help):
        if self.is_docs_build:
            return _QUALIFIER_FMT % (_help, _('Network version 2 only'))
        return _help

    def enhance_help_nova_network(self, _help):
        if self.is_docs_build:
            return _QUALIFIER_FMT % (_help, _('Compute version 2 only'))
        return _help

    @staticmethod
    def split_help(network_help, compute_help):
        return '*%(network_qualifier)s:*\n  %(network_help)s\n\n*%(compute_qualifier)s:*\n  %(compute_help)s' % dict(network_qualifier=_('Network version 2'), network_help=network_help, compute_qualifier=_('Compute version 2'), compute_help=compute_help)

    def get_parser(self, prog_name):
        LOG.debug('get_parser(%s)', prog_name)
        parser = super(NetDetectionMixin, self).get_parser(prog_name)
        parser = self.update_parser_common(parser)
        LOG.debug('common parser: %s', parser)
        if self.is_neutron or self.is_docs_build:
            parser = self.update_parser_network(parser)
        if self.is_nova_network or self.is_docs_build:
            parser = self.update_parser_compute(parser)
        return parser

    def update_parser_common(self, parser):
        """Default is no updates to parser."""
        return parser

    def update_parser_network(self, parser):
        """Default is no updates to parser."""
        return parser

    def update_parser_compute(self, parser):
        """Default is no updates to parser."""
        return parser

    def take_action(self, parsed_args):
        if self.is_neutron:
            return self.take_action_network(self.app.client_manager.network, parsed_args)
        elif self.is_nova_network:
            return self.take_action_compute(self.app.client_manager.compute, parsed_args)

    def take_action_network(self, client, parsed_args):
        """Override to do something useful."""
        pass

    def take_action_compute(self, client, parsed_args):
        """Override to do something useful."""
        pass