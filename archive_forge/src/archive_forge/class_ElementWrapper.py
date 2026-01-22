from suds import *
from suds.mx import *
from suds.sudsobject import Object, Property
from suds.sax.element import Element
from suds.sax.text import Text
from suds.xsd.sxbasic import Attribute
class ElementWrapper(Element):
    """
    Element wrapper.
    """

    def __init__(self, content):
        Element.__init__(self, content.name, content.parent)
        self.__content = content

    def str(self, indent=0):
        return self.__content.str(indent)