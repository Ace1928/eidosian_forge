from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
class ContentTypeHeader(ParameterizedMIMEHeader):
    value_parser = staticmethod(parser.parse_content_type_header)

    def init(self, *args, **kw):
        super().init(*args, **kw)
        self._maintype = utils._sanitize(self._parse_tree.maintype)
        self._subtype = utils._sanitize(self._parse_tree.subtype)

    @property
    def maintype(self):
        return self._maintype

    @property
    def subtype(self):
        return self._subtype

    @property
    def content_type(self):
        return self.maintype + '/' + self.subtype