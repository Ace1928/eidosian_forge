from suds import *
from suds.mx import *
from suds.sudsobject import Object, Property
from suds.sax.element import Element
from suds.sax.text import Text
from suds.xsd.sxbasic import Attribute
class ObjectAppender(Appender):
    """
    An L{Object} appender.
    """

    def append(self, parent, content):
        object = content.value
        child = self.node(content)
        parent.append(child)
        for item in object:
            cont = Content(tag=item[0], value=item[1])
            Appender.append(self, child, cont)