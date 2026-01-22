import time
from oslo_log import log as logging
from os_win import _utils
import os_win.conf
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
@_utils.not_found_decorator()
@_utils.retry_decorator(exceptions=exceptions.HyperVException)
def add_multiple_virt_features(self, virt_features, parent):
    job_path, out_set_data, ret_val = self._vs_man_svc.AddFeatureSettings(parent.path_(), [f.GetText_(1) for f in virt_features])
    self.check_ret_val(ret_val, job_path)