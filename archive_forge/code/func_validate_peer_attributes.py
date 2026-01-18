from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
def validate_peer_attributes(parsed_args):
    validators.validate_int_range(parsed_args, 'remote_as', neutronv20.bgp.speaker.MIN_AS_NUM, neutronv20.bgp.speaker.MAX_AS_NUM)
    if parsed_args.auth_type != 'none' and parsed_args.password is None:
        raise exceptions.CommandError(_('Must provide password if auth-type is specified.'))
    if parsed_args.auth_type == 'none' and parsed_args.password:
        raise exceptions.CommandError(_('Must provide auth-type if password is specified.'))