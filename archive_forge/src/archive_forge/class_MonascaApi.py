from osc_lib.api import api
from monascaclient.v2_0 import alarm_definitions as ad
from monascaclient.v2_0 import alarms
from monascaclient.v2_0 import metrics
from monascaclient.v2_0 import notifications
from monascaclient.v2_0 import notificationtypes as nt
class MonascaApi(api.BaseAPI):
    SERVICE_TYPE = 'monitoring'