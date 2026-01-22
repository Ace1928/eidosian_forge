import weakref
from ..Qt import QtWidgets
from .Container import Container, HContainer, TContainer, VContainer
from .Dock import Dock
from .DockDrop import DockDrop
def addContainer(self, typ, obj):
    """Add a new container around obj"""
    new = self.makeContainer(typ)
    container = self.getContainer(obj)
    container.insert(new, 'before', obj)
    if obj is not None:
        new.insert(obj)
    self.dockdrop.raiseOverlay()
    return new