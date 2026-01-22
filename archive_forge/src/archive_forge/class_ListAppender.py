from suds import *
from suds.mx import *
from suds.sudsobject import Object, Property
from suds.sax.element import Element
from suds.sax.text import Text
from suds.xsd.sxbasic import Attribute
class ListAppender(Appender):
    """
    A list/tuple appender.
    """

    def append(self, parent, content):
        collection = content.value
        if len(collection):
            self.suspend(content)
            for item in collection:
                cont = Content(tag=content.tag, value=item)
                Appender.append(self, parent, cont)
            self.resume(content)