import copy
import io
import json
import testtools
from urllib import parse
from glanceclient.v2 import schemas
class FakeSchemaAPI(FakeAPI):

    def get(self, *args, **kwargs):
        _, raw_schema = self._request('GET', *args, **kwargs)
        return schemas.Schema(raw_schema)