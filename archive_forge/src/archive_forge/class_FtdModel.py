from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import urlparse
class FtdModel:
    FTD_ASA5506_X = 'Cisco ASA5506-X Threat Defense'
    FTD_ASA5508_X = 'Cisco ASA5508-X Threat Defense'
    FTD_ASA5516_X = 'Cisco ASA5516-X Threat Defense'
    FTD_2110 = 'Cisco Firepower 2110 Threat Defense'
    FTD_2120 = 'Cisco Firepower 2120 Threat Defense'
    FTD_2130 = 'Cisco Firepower 2130 Threat Defense'
    FTD_2140 = 'Cisco Firepower 2140 Threat Defense'

    @classmethod
    def supported_models(cls):
        return [getattr(cls, item) for item in dir(cls) if item.startswith('FTD_')]