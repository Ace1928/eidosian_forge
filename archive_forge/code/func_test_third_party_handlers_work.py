import logging
import os
import pathlib
import sys
import time
import pytest
@pytest.mark.logmodepython
@pytest.mark.logmodemixed
def test_third_party_handlers_work():
    simulate_evacuation()