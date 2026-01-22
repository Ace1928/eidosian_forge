import traceback
import lxml.etree
import ncclient
from os_ken.base import app_manager
from os_ken.lib.netconf import constants as nc_consts
from os_ken.lib import hub
from os_ken.lib import of_config
from os_ken.lib.of_config import capable_switch
from os_ken.lib.of_config import constants as ofc_consts
def _do_of_config(self):
    self._do_get()
    self._do_get_config('running')
    self._do_get_config('startup')
    try:
        self._do_get_config('candidate')
    except ncclient.NCClientError:
        traceback.print_exc()
    self._do_edit_config(SWITCH_PORT_DOWN)
    self._do_edit_config(SWITCH_ADVERTISED)
    self._do_edit_config(SWITCH_CONTROLLER)
    self._set_ports_down()
    self.switch.close_session()