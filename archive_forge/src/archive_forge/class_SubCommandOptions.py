from twisted.python import usage
from twisted.trial import unittest
class SubCommandOptions(usage.Options):
    optFlags = [('europian-swallow', None, 'set default swallow type to Europian')]
    subCommands = [('inquisition', 'inquest', InquisitionOptions, 'Perform an inquisition'), ('holyquest', 'quest', HolyQuestOptions, 'Embark upon a holy quest')]