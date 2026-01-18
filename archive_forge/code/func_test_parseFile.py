from twisted.runner import inetdconf
from twisted.trial import unittest
def test_parseFile(self) -> None:
    """
        Services are parsed from given C{file}.
        """
    conf = inetdconf.ServicesConf()
    with open(self.servicesFilename2) as f:
        conf.parseFile(f)
    self.assertEqual(conf.services, {('https', 'tcp'): 443})