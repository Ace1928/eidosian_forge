import os
import sqlite3
from io import BytesIO
from os.path import dirname
from os.path import join as pjoin
from ..testing import suppress_warnings
import unittest
import pytest
from .. import nifti1
from ..optpkg import optional_package
def test_storage_instances(db):
    studies = dft.get_studies(data_dir)
    sis = studies[0].series[0].storage_instances
    assert len(sis) == 2
    assert sis[0].instance_number == 1
    assert sis[1].instance_number == 2
    assert sis[0].uid == '1.3.12.2.1107.5.2.32.35119.2010011420300180088599504.0'
    assert sis[1].uid == '1.3.12.2.1107.5.2.32.35119.2010011420300180088599504.1'