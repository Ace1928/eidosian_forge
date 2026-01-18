import os
import mercantile
import pytest
import requests
import xyzservices.providers as xyz
@pytest.mark.request
@pytest.mark.parametrize('provider_name', xyz.OrdnanceSurvey)
def test_os(provider_name):
    try:
        token = os.environ['ORDNANCESURVEY']
    except KeyError:
        pytest.xfail('Missing API token.')
    if token == '':
        pytest.xfail('Token empty.')
    provider = xyz.OrdnanceSurvey[provider_name](key=token)
    get_test_result(provider, allow_403=False)