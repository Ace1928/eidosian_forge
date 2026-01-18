from twisted.python import usage
from twisted.trial import unittest
def test_subcommandWithFlagsAndOptions(self):
    """
        Flags and options of a subcommand are assigned.
        """
    o = SubCommandOptions()
    o.parseOptions(['inquisition', '--expect', '--torture-device=feather'])
    self.assertFalse(o['europian-swallow'])
    self.assertEqual(o.subCommand, 'inquisition')
    self.assertIsInstance(o.subOptions, InquisitionOptions)
    self.assertTrue(o.subOptions['expect'])
    self.assertEqual(o.subOptions['torture-device'], 'feather')