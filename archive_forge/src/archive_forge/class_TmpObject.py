from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
class TmpObject(base.Resource):
    id = '4'