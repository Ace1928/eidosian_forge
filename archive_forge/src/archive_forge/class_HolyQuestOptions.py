from twisted.python import usage
from twisted.trial import unittest
class HolyQuestOptions(usage.Options):
    optFlags = [('horseback', 'h', 'use a horse'), ('for-grail', 'g')]