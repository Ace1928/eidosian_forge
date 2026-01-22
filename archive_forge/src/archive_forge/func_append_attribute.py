from suds import *
from suds.umx import *
from suds.umx.core import Core
from suds.resolver import NodeResolver, Frame
from suds.sudsobject import Factory
from logging import getLogger
def append_attribute(self, name, value, content):
    """
        Append an attribute name/value into L{Content.data}.
        @param name: The attribute name
        @type name: basestring
        @param value: The attribute's value
        @type value: basestring
        @param content: The current content being unmarshalled.
        @type content: L{Content}
        """
    type = self.resolver.findattr(name)
    if type is None:
        log.warning('attribute (%s) type, not-found', name)
    else:
        value = self.translated(value, type)
    Core.append_attribute(self, name, value, content)