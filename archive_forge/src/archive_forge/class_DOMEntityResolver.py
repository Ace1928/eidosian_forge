import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
class DOMEntityResolver(object):
    __slots__ = ('_opener',)

    def resolveEntity(self, publicId, systemId):
        assert systemId is not None
        source = DOMInputSource()
        source.publicId = publicId
        source.systemId = systemId
        source.byteStream = self._get_opener().open(systemId)
        source.encoding = self._guess_media_encoding(source)
        import posixpath, urllib.parse
        parts = urllib.parse.urlparse(systemId)
        scheme, netloc, path, params, query, fragment = parts
        if path and (not path.endswith('/')):
            path = posixpath.dirname(path) + '/'
            parts = (scheme, netloc, path, params, query, fragment)
            source.baseURI = urllib.parse.urlunparse(parts)
        return source

    def _get_opener(self):
        try:
            return self._opener
        except AttributeError:
            self._opener = self._create_opener()
            return self._opener

    def _create_opener(self):
        import urllib.request
        return urllib.request.build_opener()

    def _guess_media_encoding(self, source):
        info = source.byteStream.info()
        if 'Content-Type' in info:
            for param in info.getplist():
                if param.startswith('charset='):
                    return param.split('=', 1)[1].lower()