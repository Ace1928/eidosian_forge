import os
import pytest
from lxml import etree
@pytest.fixture
def voila_resources(show_handles):
    return {'gridstack': {'show_handles': show_handles}}