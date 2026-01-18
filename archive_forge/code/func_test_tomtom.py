import os
import mercantile
import pytest
import requests
import xyzservices.providers as xyz
@pytest.mark.request
@pytest.mark.parametrize('provider_name', xyz.TomTom)
def test_tomtom(provider_name):
    try:
        token = os.environ['TOMTOM']
    except KeyError:
        pytest.xfail('Missing API token.')
    if token == '':
        pytest.xfail('Token empty.')
    provider = xyz.TomTom[provider_name](apikey=token)
    get_test_result(provider, allow_403=False)