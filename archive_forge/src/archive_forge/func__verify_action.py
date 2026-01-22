import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def _verify_action(self, actions, type_, name, value):
    try:
        action = actions[0]
        if action.cls_action_type != type_:
            return 'Action type error. send:%s, val:%s' % (type_, action.cls_action_type)
    except IndexError:
        return 'Action is not setting.'
    f_value = None
    if name:
        try:
            if isinstance(name, list):
                f_value = [getattr(action, n) for n in name]
            else:
                f_value = getattr(action, name)
        except AttributeError:
            pass
    if f_value != value:
        return 'Value error. send:%s=%s val:%s' % (name, value, f_value)
    return True