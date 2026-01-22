import functools
import re
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win._i18n import _
from os_win import conf
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
def _bind_security_rules(self, port, sg_rules):
    acls = _wqlutils.get_element_associated_class(self._conn, self._PORT_EXT_ACL_SET_DATA, element_instance_id=port.InstanceID)
    add_acls = []
    processed_sg_rules = []
    weights = self._get_new_weights(sg_rules, acls)
    index = 0
    for sg_rule in sg_rules:
        filtered_acls = self._filter_security_acls(sg_rule, acls)
        if filtered_acls:
            continue
        acl = self._create_security_acl(sg_rule, weights[index])
        add_acls.append(acl)
        index += 1
        processed_sg_rules.append(sg_rule)
    if add_acls:
        self._jobutils.add_multiple_virt_features(add_acls, port)
        acls.extend(processed_sg_rules)