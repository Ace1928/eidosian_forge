from suds import *
from suds.sudsobject import *
from suds.xsd import qualify, isqref
from suds.xsd.sxbuiltin import Factory
from logging import getLogger
class ElementQuery(Query):
    """
    Schema query class that searches for Element references in the specified
    schema. Matches on root Elements by qname first, then searches deeper into
    the document.

    """

    def execute(self, schema):
        result = schema.elements.get(self.ref)
        if self.filter(result):
            result = self.__deepsearch(schema)
        return self.result(result)

    def __deepsearch(self, schema):
        from suds.xsd.sxbasic import Element
        result = None
        for e in schema.all:
            result = e.find(self.ref, (Element,))
            if self.filter(result):
                result = None
            else:
                break
        return result