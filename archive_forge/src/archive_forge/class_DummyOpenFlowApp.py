from os_ken.base import app_manager
from os_ken.ofproto import ofproto_v1_3
class DummyOpenFlowApp(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]