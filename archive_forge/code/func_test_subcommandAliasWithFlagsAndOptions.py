from twisted.python import usage
from twisted.trial import unittest
def test_subcommandAliasWithFlagsAndOptions(self):
    """
        Flags and options of a subcommand alias are assigned.
        """
    o = SubCommandOptions()
    o.parseOptions(['inquest', '--expect', '--torture-device=feather'])
    self.assertFalse(o['europian-swallow'])
    self.assertEqual(o.subCommand, 'inquisition')
    self.assertIsInstance(o.subOptions, InquisitionOptions)
    self.assertTrue(o.subOptions['expect'])
    self.assertEqual(o.subOptions['torture-device'], 'feather')