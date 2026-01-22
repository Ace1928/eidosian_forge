import logging
import numbers
from os_ken.lib import ip
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv4
from os_ken.services.protocols.bgp.utils.validation import is_valid_asn
from os_ken.services.protocols.bgp import rtconf
from os_ken.services.protocols.bgp.rtconf.base import BaseConf
from os_ken.services.protocols.bgp.rtconf.base import BaseConfListener
from os_ken.services.protocols.bgp.rtconf.base import compute_optional_conf
from os_ken.services.protocols.bgp.rtconf.base import ConfigTypeError
from os_ken.services.protocols.bgp.rtconf.base import ConfigValueError
from os_ken.services.protocols.bgp.rtconf.base import MissingRequiredConf
from os_ken.services.protocols.bgp.rtconf.base import validate
class CommonConfListener(BaseConfListener):
    """Base listener for various changes to common configurations."""

    def __init__(self, global_conf):
        super(CommonConfListener, self).__init__(global_conf)
        global_conf.add_listener(CommonConf.CONF_CHANGED_EVT, self.on_update_common_conf)

    def on_update_common_conf(self, evt):
        raise NotImplementedError('This method should be overridden.')