import weakref
from ..Qt import QtWidgets
from .Container import Container, HContainer, TContainer, VContainer
from .Dock import Dock
from .DockDrop import DockDrop
def addDock(self, dock=None, position='bottom', relativeTo=None, **kwds):
    """Adds a dock to this area.
        
        ============== =================================================================
        **Arguments:**
        dock           The new Dock object to add. If None, then a new Dock will be 
                       created.
        position       'bottom', 'top', 'left', 'right', 'above', or 'below'
        relativeTo     If relativeTo is None, then the new Dock is added to fill an 
                       entire edge of the window. If relativeTo is another Dock, then 
                       the new Dock is placed adjacent to it (or in a tabbed 
                       configuration for 'above' and 'below'). 
        ============== =================================================================
        
        All extra keyword arguments are passed to Dock.__init__() if *dock* is
        None.        
        """
    if dock is None:
        dock = Dock(**kwds)
    if not self.temporary:
        dock.orig_area = self
    if relativeTo is None or relativeTo is self:
        if self.topContainer is None:
            container = self
            neighbor = None
        else:
            container = self.topContainer
            neighbor = None
    else:
        if isinstance(relativeTo, str):
            relativeTo = self.docks[relativeTo]
        container = self.getContainer(relativeTo)
        if container is None:
            raise TypeError('Dock %s is not contained in a DockArea; cannot add another dock relative to it.' % relativeTo)
        neighbor = relativeTo
    neededContainer = {'bottom': 'vertical', 'top': 'vertical', 'left': 'horizontal', 'right': 'horizontal', 'above': 'tab', 'below': 'tab'}[position]
    if neededContainer != container.type() and container.type() == 'tab':
        neighbor = container
        container = container.container()
    if neededContainer != container.type():
        if neighbor is None:
            container = self.addContainer(neededContainer, self.topContainer)
        else:
            container = self.addContainer(neededContainer, neighbor)
    insertPos = {'bottom': 'after', 'top': 'before', 'left': 'before', 'right': 'after', 'above': 'before', 'below': 'after'}[position]
    old = dock.container()
    container.insert(dock, insertPos, neighbor)
    self.docks[dock.name()] = dock
    if old is not None:
        old.apoptose()
    return dock