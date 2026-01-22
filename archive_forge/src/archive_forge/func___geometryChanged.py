from ..Point import Point
def __geometryChanged(self):
    if self.__parent is None:
        return
    if self.__itemAnchor is None:
        return
    o = self.mapToParent(Point(0, 0))
    a = self.boundingRect().bottomRight() * Point(self.__itemAnchor)
    a = self.mapToParent(a)
    p = self.__parent.boundingRect().bottomRight() * Point(self.__parentAnchor)
    off = Point(self.__offset)
    pos = p + (o - a) + off
    self.setPos(pos)