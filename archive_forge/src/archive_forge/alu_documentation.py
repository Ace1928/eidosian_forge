from lxml import etree
from .default import DefaultDeviceHandler
from ncclient.operations.third_party.alu.rpc import GetConfiguration, LoadConfiguration, ShowCLI
from ncclient.xml_ import BASE_NS_1_0

    Alcatel-Lucent 7x50 handler for device specific information.
    