import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore
class HoverEvent(object):
    """
    Instances of this class are delivered to items in a :class:`GraphicsScene <pyqtgraph.GraphicsScene>` via their hoverEvent() method when the mouse is hovering over the item.
    This event class both informs items that the mouse cursor is nearby and allows items to 
    communicate with one another about whether each item will accept *potential* mouse events. 
    
    It is common for multiple overlapping items to receive hover events and respond by changing 
    their appearance. This can be misleading to the user since, in general, only one item will
    respond to mouse events. To avoid this, items make calls to event.acceptClicks(button) 
    and/or acceptDrags(button).
    
    Each item may make multiple calls to acceptClicks/Drags, each time for a different button. 
    If the method returns True, then the item is guaranteed to be
    the recipient of the claimed event IF the user presses the specified mouse button before
    moving. If claimEvent returns False, then this item is guaranteed NOT to get the specified
    event (because another has already claimed it) and the item should change its appearance 
    accordingly.
    
    event.isEnter() returns True if the mouse has just entered the item's shape;
    event.isExit() returns True if the mouse has just left.
    """

    def __init__(self, moveEvent, acceptable):
        self.enter = False
        self.acceptable = acceptable
        self.exit = False
        self.__clickItems = weakref.WeakValueDictionary()
        self.__dragItems = weakref.WeakValueDictionary()
        self.currentItem = None
        if moveEvent is not None:
            self._scenePos = moveEvent.scenePos()
            self._screenPos = moveEvent.screenPos()
            self._lastScenePos = moveEvent.lastScenePos()
            self._lastScreenPos = moveEvent.lastScreenPos()
            self._buttons = moveEvent.buttons()
            self._modifiers = moveEvent.modifiers()
        else:
            self.exit = True

    def isEnter(self):
        """Returns True if the mouse has just entered the item's shape"""
        return self.enter

    def isExit(self):
        """Returns True if the mouse has just exited the item's shape"""
        return self.exit

    def acceptClicks(self, button):
        """Inform the scene that the item (that the event was delivered to)
        would accept a mouse click event if the user were to click before
        moving the mouse again.
        
        Returns True if the request is successful, otherwise returns False (indicating
        that some other item would receive an incoming click).
        """
        if not self.acceptable:
            return False
        if button not in self.__clickItems:
            self.__clickItems[button] = self.currentItem
            return True
        return False

    def acceptDrags(self, button):
        """Inform the scene that the item (that the event was delivered to)
        would accept a mouse drag event if the user were to drag before
        the next hover event.
        
        Returns True if the request is successful, otherwise returns False (indicating
        that some other item would receive an incoming drag event).
        """
        if not self.acceptable:
            return False
        if button not in self.__dragItems:
            self.__dragItems[button] = self.currentItem
            return True
        return False

    def scenePos(self):
        """Return the current scene position of the mouse."""
        return Point(self._scenePos)

    def screenPos(self):
        """Return the current screen position of the mouse."""
        return Point(self._screenPos)

    def lastScenePos(self):
        """Return the previous scene position of the mouse."""
        return Point(self._lastScenePos)

    def lastScreenPos(self):
        """Return the previous screen position of the mouse."""
        return Point(self._lastScreenPos)

    def buttons(self):
        """
        Return the buttons currently pressed on the mouse.
        (see QGraphicsSceneMouseEvent::buttons in the Qt documentation)
        """
        return self._buttons

    def pos(self):
        """
        Return the current position of the mouse in the coordinate system of the item
        that the event was delivered to.
        """
        return Point(self.currentItem.mapFromScene(self._scenePos))

    def lastPos(self):
        """
        Return the previous position of the mouse in the coordinate system of the item
        that the event was delivered to.
        """
        return Point(self.currentItem.mapFromScene(self._lastScenePos))

    def __repr__(self):
        if self.exit:
            return '<HoverEvent exit=True>'
        if self.currentItem is None:
            lp = self._lastScenePos
            p = self._scenePos
        else:
            lp = self.lastPos()
            p = self.pos()
        return '<HoverEvent (%g,%g)->(%g,%g) buttons=%s enter=%s exit=%s>' % (lp.x(), lp.y(), p.x(), p.y(), str(self.buttons()), str(self.isEnter()), str(self.isExit()))

    def modifiers(self):
        """Return any keyboard modifiers currently pressed.
        (see QGraphicsSceneMouseEvent::modifiers in the Qt documentation)        
        """
        return self._modifiers

    def clickItems(self):
        return self.__clickItems

    def dragItems(self):
        return self.__dragItems