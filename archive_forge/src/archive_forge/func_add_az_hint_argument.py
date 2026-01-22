from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
def add_az_hint_argument(parser, resource):
    parser.add_argument('--availability-zone-hint', metavar='AVAILABILITY_ZONE', action='append', dest='availability_zone_hints', help=_('Availability Zone for the %s (requires availability zone extension, this option can be repeated).') % resource)