from libcloud.compute.providers import Provider
from libcloud.common.dimensiondata import API_ENDPOINTS, DimensionDataConnection
from libcloud.compute.drivers.dimensiondata import DimensionDataNodeDriver
class NTTAmericaNodeDriver(DimensionDataNodeDriver):
    """
    NTT America node driver, based on Dimension Data driver
    """
    selected_region = None
    connectionCls = DimensionDataConnection
    name = 'NTTAmerica'
    website = 'http://www.nttamerica.com/'
    type = Provider.NTTA
    features = {'create_node': ['password']}
    api_version = 1.0

    def __init__(self, key, secret=None, secure=True, host=None, port=None, api_version=None, region=DEFAULT_REGION, **kwargs):
        if region not in API_ENDPOINTS:
            raise ValueError('Invalid region: %s' % region)
        self.selected_region = API_ENDPOINTS[region]
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, api_version=api_version, region=region, **kwargs)