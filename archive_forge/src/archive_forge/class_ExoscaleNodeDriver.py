from libcloud.compute.providers import Provider
from libcloud.compute.drivers.cloudstack import CloudStackNodeDriver
class ExoscaleNodeDriver(CloudStackNodeDriver):
    type = Provider.EXOSCALE
    name = 'Exoscale'
    website = 'https://www.exoscale.com/'
    host = 'api.exoscale.com'
    path = '/compute'