import os
import mercantile
import pytest
import requests
import xyzservices.providers as xyz
@pytest.mark.request
@pytest.mark.parametrize('provider_name', xyz.Thunderforest)
def test_thunderforest(provider_name):
    try:
        token = os.environ['THUNDERFOREST']
    except KeyError:
        pytest.xfail('Missing API token.')
    if token == '':
        pytest.xfail('Token empty.')
    provider = xyz.Thunderforest[provider_name](apikey=token)
    get_test_result(provider, allow_403=False)