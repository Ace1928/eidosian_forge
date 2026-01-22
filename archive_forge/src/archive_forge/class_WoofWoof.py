import inspect
import warnings
import debtcollector
from debtcollector.fixtures import disable
from debtcollector import moves
from debtcollector import removals
from debtcollector import renames
from debtcollector.tests import base as test_base
from debtcollector import updating
class WoofWoof(object):

    @property
    def bark(self):
        return 'woof'

    @property
    @moves.moved_property('bark')
    def burk(self):
        return self.bark

    @property
    @moves.moved_property('bark', category=PendingDeprecationWarning)
    def berk(self):
        return self.bark

    @removals.removed_kwarg('resp', message="Please use 'response' instead")
    @classmethod
    def factory(cls, resp=None, response=None):
        return 'super-duper'