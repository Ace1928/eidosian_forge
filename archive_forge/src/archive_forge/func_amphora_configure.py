import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def amphora_configure(self, amphora_id):
    """Update the amphora agent configuration

        :param string amphora_id:
            ID of the amphora to configure
        :return:
            Response Code from the API
        """
    url = const.BASE_AMPHORA_CONFIGURE_URL.format(uuid=amphora_id)
    response = self._create(url, method='PUT')
    return response