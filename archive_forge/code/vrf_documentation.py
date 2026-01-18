import logging
import pprint
from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_ERROR
from os_ken.services.protocols.bgp.operator.command import STATUS_OK
from os_ken.services.protocols.bgp.operator.commands.responses import \
from os_ken.services.protocols.bgp.operator.views.conf import ConfDetailView
from os_ken.services.protocols.bgp.operator.views.conf import ConfDictView
from .route_formatter_mixin import RouteFormatterMixin
Main node for vrf related commands. Acts also as Routes node (that's why
    it inherits from it) for legacy reasons.
    