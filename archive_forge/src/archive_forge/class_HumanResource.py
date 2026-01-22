from unittest import mock
from oslotest import base as test_base
from ironicclient.common.apiclient import base
class HumanResource(base.Resource):
    HUMAN_ID = True