import os
import pyomo.common.unittest as unittest
from pyomo.common.env import CtypesEnviron
def test_temp_env_str(self):
    orig_env = CtypesEnviron()
    orig_env_has_1 = 'TEST_ENV_1' in orig_env
    orig_env_has_2 = 'TEST_ENV_2' in orig_env
    if 'TEST_ENV_1' in orig_env:
        del orig_env['TEST_ENV_1']
    orig_env['TEST_ENV_2'] = 'test value: 2'
    self.assertIsNone(os.environ.get('TEST_ENV_1', None))
    self.assertEqual(os.environ['TEST_ENV_2'], 'test value: 2')
    for interface in orig_env.interfaces:
        self.assertIsNone(interface.dll.wgetenv(u'TEST_ENV_1'))
        self.assertIsNone(interface.dll.getenv(b'TEST_ENV_1'))
        self.assertEqual(interface.dll.wgetenv(u'TEST_ENV_2'), u'test value: 2')
        self.assertEqual(interface.dll.getenv(b'TEST_ENV_2'), b'test value: 2')
    with CtypesEnviron(TEST_ENV_1='test value: 1') as env:
        self.assertEqual(os.environ['TEST_ENV_1'], 'test value: 1')
        self.assertEqual(os.environ['TEST_ENV_2'], 'test value: 2')
        for interface in env.interfaces:
            self.assertEqual(interface.dll.wgetenv(u'TEST_ENV_1'), u'test value: 1')
            self.assertEqual(interface.dll.getenv(b'TEST_ENV_1'), b'test value: 1')
            self.assertEqual(interface.dll.wgetenv(u'TEST_ENV_2'), u'test value: 2')
            self.assertEqual(interface.dll.getenv(b'TEST_ENV_2'), b'test value: 2')
        del env['TEST_ENV_2']
        self.assertIsNone(os.environ.get('TEST_ENV_2', None))
        for interface in env.interfaces:
            self.assertIsNone(interface.dll.wgetenv(u'TEST_ENV_2'))
            self.assertIsNone(interface.dll.getenv(b'TEST_ENV_2'))
    self.assertIsNone(os.environ.get('TEST_ENV_1', None))
    self.assertEqual(os.environ['TEST_ENV_2'], 'test value: 2')
    for interface in orig_env.interfaces:
        self.assertIsNone(interface.dll.wgetenv(u'TEST_ENV_1'))
        self.assertIsNone(interface.dll.getenv(b'TEST_ENV_1'))
        self.assertEqual(interface.dll.wgetenv(u'TEST_ENV_2'), u'test value: 2')
        self.assertEqual(interface.dll.getenv(b'TEST_ENV_2'), b'test value: 2')
    orig_env.restore()
    self.assertEqual(orig_env_has_1, 'TEST_ENV_1' in os.environ)
    self.assertEqual(orig_env_has_2, 'TEST_ENV_2' in os.environ)