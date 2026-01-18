from __future__ import absolute_import, division, print_function
import traceback
from random import randint
from ansible.module_utils.common.network import is_mac
from ansible.module_utils.basic import missing_required_lib
def update_nvdimm_config(self, nvdimm_device, nvdimm_size):
    nvdimm_spec = vim.vm.device.VirtualDeviceSpec()
    nvdimm_spec.operation = vim.vm.device.VirtualDeviceSpec.Operation.edit
    nvdimm_spec.device = nvdimm_device
    nvdimm_device.capacityInMB = nvdimm_size
    return nvdimm_spec