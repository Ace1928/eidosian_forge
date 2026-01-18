import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
def test_update_config_fail_message(self):
    """When failing, the __force_dict__ key is suggested"""
    from pecan import configuration
    bad_dict = {'bad name': 'value'}
    try:
        configuration.Config(bad_dict)
    except ValueError as error:
        assert "consider using the '__force_dict__'" in str(error)