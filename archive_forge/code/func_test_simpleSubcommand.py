from twisted.python import usage
from twisted.trial import unittest
def test_simpleSubcommand(self):
    """
        A subcommand is recognized.
        """
    o = SubCommandOptions()
    o.parseOptions(['--europian-swallow', 'inquisition'])
    self.assertTrue(o['europian-swallow'])
    self.assertEqual(o.subCommand, 'inquisition')
    self.assertIsInstance(o.subOptions, InquisitionOptions)
    self.assertFalse(o.subOptions['expect'])
    self.assertEqual(o.subOptions['torture-device'], 'comfy-chair')