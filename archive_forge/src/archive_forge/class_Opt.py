from twisted.python import usage
from twisted.trial import unittest
class Opt(usage.Options):
    subCommands = [('foo', 'f', SubOpt, 'bar')]