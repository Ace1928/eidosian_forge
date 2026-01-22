from suds import *
from suds.mx import *
from suds.sudsobject import Object, Property
from suds.sax.element import Element
from suds.sax.text import Text
from suds.xsd.sxbasic import Attribute
class ContentAppender:
    """
    Appender used to add content to marshalled objects.
    @ivar default: The default appender.
    @type default: L{Appender}
    @ivar appenders: A I{table} of appenders mapped by class.
    @type appenders: I{table}
    """

    def __init__(self, marshaller):
        """
        @param marshaller: A marshaller.
        @type marshaller: L{suds.mx.core.Core}
        """
        self.default = PrimitiveAppender(marshaller)
        self.appenders = ((Matcher(None), NoneAppender(marshaller)), (Matcher(null), NoneAppender(marshaller)), (Matcher(Property), PropertyAppender(marshaller)), (Matcher(Object), ObjectAppender(marshaller)), (Matcher(Element), ElementAppender(marshaller)), (Matcher(Text), TextAppender(marshaller)), (Matcher(list), ListAppender(marshaller)), (Matcher(tuple), ListAppender(marshaller)))

    def append(self, parent, content):
        """
        Select an appender and append the content to parent.
        @param parent: A parent node.
        @type parent: L{Element}
        @param content: The content to append.
        @type content: L{Content}
        """
        appender = self.default
        for matcher, candidate_appender in self.appenders:
            if matcher == content.value:
                appender = candidate_appender
                break
        appender.append(parent, content)