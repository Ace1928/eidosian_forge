import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
def _check_type_parameters(attrs, type, is_create):
    req_params = MANDATORY_PARAMETERS[type]
    opt_params = OPTIONAL_PARAMETERS[type]
    type_params = req_params | opt_params
    notreq_params = set(itertools.chain(*[v for k, v in MANDATORY_PARAMETERS.items() if k != type]))
    notreq_params -= type_params
    if is_create and None in map(attrs.get, req_params):
        msg = _('"Create" rule command for type "%(rule_type)s" requires arguments: %(args)s') % {'rule_type': type, 'args': ', '.join(sorted(req_params))}
        raise exceptions.CommandError(msg)
    if set(attrs.keys()) & notreq_params:
        msg = _('Rule type "%(rule_type)s" only requires arguments: %(args)s') % {'rule_type': type, 'args': ', '.join(sorted(type_params))}
        raise exceptions.CommandError(msg)