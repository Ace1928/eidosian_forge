from ncclient.operations.third_party.huawei.rpc import *
from ncclient.xml_ import BASE_NS_1_0
from .default import DefaultDeviceHandler

    Huawei handler for device specific information.

    In the device_params dictionary, which is passed to __init__, you can specify
    the parameter "ssh_subsystem_name". That allows you to configure the preferred
    SSH subsystem name that should be tried on your Huawei switch. If connecting with
    that name fails, or you didn't specify that name, the other known subsystem names
    will be tried. However, if you specify it then this name will be tried first.

    