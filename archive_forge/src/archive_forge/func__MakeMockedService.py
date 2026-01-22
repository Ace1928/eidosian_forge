import difflib
import sys
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import encoding
from apitools.base.py import exceptions
def _MakeMockedService(api_name, collection_name, mock_client, service, real_service):

    class MockedService(base_api.BaseApiService):
        pass
    for method in service.GetMethodsList():
        real_method = None
        if real_service:
            real_method = getattr(real_service, method)
        setattr(MockedService, method, _MockedMethod(api_name + '.' + collection_name + '.' + method, mock_client, real_method))
    return MockedService