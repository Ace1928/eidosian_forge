import re
import string
import weakref
from string import whitespace
from types import MethodType
from .constants import DefaultValue
from .trait_base import Undefined, Uninitialized
from .trait_errors import TraitError
from .trait_notifiers import TraitChangeNotifyWrapper
from .util.weakiddict import WeakIDKeyDict
class ListenerNotifyWrapper(TraitChangeNotifyWrapper):

    def __init__(self, handler, owner, id, listener, target=None):
        self.type = ListenerType.get(self.init(handler, weakref.ref(owner, self.owner_deleted), target))
        self.id = id
        self.listener = listener

    def listener_deleted(self, ref):
        owner = self.owner()
        if owner is not None:
            dict = owner.__dict__.get(TraitsListener)
            listeners = dict.get(self.id)
            listeners.remove(self)
            if len(listeners) == 0:
                del dict[self.id]
                if len(dict) == 0:
                    del owner.__dict__[TraitsListener]
                self.listener.unregister(owner)
        self.object = self.owner = self.listener = None

    def owner_deleted(self, ref):
        self.object = self.owner = None