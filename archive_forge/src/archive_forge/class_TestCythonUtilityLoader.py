import unittest
from Cython.Compiler import Code, UtilityCode
class TestCythonUtilityLoader(TestTempitaUtilityLoader):
    """
    Test loading CythonUtilityCodes
    """
    expected = (None, 'test {{cy_loader}} impl')
    expected_tempita = (None, 'test CyLoader impl')
    required = (None, 'req {{cy_loader}} impl')
    required_tempita = (None, 'req CyLoader impl')
    context = dict(cy_loader='CyLoader')
    name = 'TestCyUtilityLoader'
    filename = 'TestCyUtilityLoader.pyx'
    cls = UtilityCode.CythonUtilityCode
    cls.proto = None
    test_load = TestUtilityLoader.test_load
    test_load_tempita = TestTempitaUtilityLoader.test_load