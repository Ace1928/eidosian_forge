import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
def test_project_parent():
    project = central.select.project('pyxnat_tests')
    assert not project.parent()