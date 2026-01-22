import testtools
from neutron_lib.hacking import checks
from neutron_lib.hacking import translation_checks as tc
from neutron_lib.tests import _base as base
def __get_msg(fun):
    msg = 'N521: jsonutils.%(fun)s must be used instead of json.%(fun)s' % {'fun': fun}
    return [(0, msg)]