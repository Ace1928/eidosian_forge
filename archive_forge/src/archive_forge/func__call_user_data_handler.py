import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def _call_user_data_handler(self, operation, src, dst):
    if hasattr(self, '_user_data'):
        for key, (data, handler) in list(self._user_data.items()):
            if handler is not None:
                handler.handle(operation, key, data, src, dst)