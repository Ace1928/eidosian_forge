import argparse
import os
import numpy as np
import pytest
from _pytest.runner import pytest_runtest_makereport as orig_pytest_runtest_makereport
import pennylane as qml
def _skip_if(dev, capabilities):
    """Skip test if device has any of the given capabilities."""
    dev_capabilities = dev.capabilities()
    for capability, value in capabilities.items():
        if capability not in dev_capabilities or dev_capabilities[capability] == value:
            pytest.skip(f'Test skipped for {dev.name} device with capability {capability}:{value}.')