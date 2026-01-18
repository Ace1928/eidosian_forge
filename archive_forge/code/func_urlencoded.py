from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
@property
def urlencoded(self):
    return urlencode(self.twotuples)