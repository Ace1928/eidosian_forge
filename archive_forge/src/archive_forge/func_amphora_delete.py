import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
@correct_return_codes
def amphora_delete(self, amphora_id):
    """Delete an amphora

        :param string amphora_id:
            The ID of the amphora to delete
        :return:
            Response Code from the API
        """
    url = const.BASE_SINGLE_AMPHORA_URL.format(uuid=amphora_id)
    response = self._delete(url)
    return response