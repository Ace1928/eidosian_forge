from __future__ import annotations
from ..runtime import driver
class TensorMapManager:

    def __init__(self):
        self.tensormaps_device = {}

    def __getitem__(self, key: tuple):
        if key in self.tensormaps_device:
            return int(self.tensormaps_device[key])
        else:
            e, args = key
            t_tensormap = e.tensormap(args)
            TENSORMAP_SIZE_IN_BYTES = 128
            t_tensormap_device = driver.utils.cuMemAlloc(TENSORMAP_SIZE_IN_BYTES)
            driver.utils.cuMemcpyHtoD(t_tensormap_device, t_tensormap, TENSORMAP_SIZE_IN_BYTES)
            self.tensormaps_device[key] = t_tensormap_device
            return int(self.tensormaps_device[key])

    def __del__(self):
        for _, v in self.tensormaps_device.items():
            driver.utils.cuMemFree(v)