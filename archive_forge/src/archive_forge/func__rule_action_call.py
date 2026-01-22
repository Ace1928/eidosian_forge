import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
def _rule_action_call(client, action, rule_type):
    rule_type = rule_type.replace('-', '_')
    func_name = '%(action)s_qos_%(rule_type)s_rule' % {'action': action, 'rule_type': rule_type}
    return getattr(client, func_name)