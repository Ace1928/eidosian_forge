from openstackclient.common import availability_zone
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def _build_volume_az_datalist(volume_az, long_datalist=False):
    datalist = ()
    if not long_datalist:
        datalist = (volume_az.name, 'available')
    else:
        datalist = (volume_az.name, 'available', '', '', '', '')
    return (datalist,)