import os
import mercantile
import pytest
import requests
import xyzservices.providers as xyz
@pytest.mark.request
@pytest.mark.parametrize('provider_name', xyz.OpenWeatherMap)
def test_openweathermap(provider_name):
    try:
        token = os.environ['OPENWEATHERMAP']
    except KeyError:
        pytest.xfail('Missing API token.')
    if token == '':
        pytest.xfail('Token empty.')
    provider = xyz.OpenWeatherMap[provider_name](apiKey=token)
    get_test_result(provider, allow_403=False)