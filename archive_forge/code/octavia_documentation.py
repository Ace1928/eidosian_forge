import functools
import urllib.parse as urlparse
from osc_lib.api import api
from osc_lib import exceptions as osc_exc
from octaviaclient.api import constants as const
from octaviaclient.api import exceptions
Delete a availabilityzone profile

        :param string availabilityzoneprofile_id:
            ID of the availabilityzone profile to delete
        :return:
            Response Code from the API
        