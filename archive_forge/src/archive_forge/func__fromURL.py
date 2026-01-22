from typing import cast
from urllib.parse import quote as urlquote, unquote as urlunquote, urlunsplit
from hyperlink import URL as _URL
@classmethod
def _fromURL(cls, urlInstance):
    """
        Reconstruct all the public instance variables of this L{URLPath} from
        its underlying L{_URL}.

        @param urlInstance: the object to base this L{URLPath} on.
        @type urlInstance: L{_URL}

        @return: a new L{URLPath}
        """
    self = cls.__new__(cls)
    self._url = urlInstance.replace(path=urlInstance.path or [''])
    self._scheme = self._url.scheme.encode('ascii')
    self._netloc = self._url.authority().encode('ascii')
    self._path = _URL(path=self._url.path, rooted=True).asURI().asText().encode('ascii')
    self._query = _URL(query=self._url.query).asURI().asText().encode('ascii')[1:]
    self._fragment = self._url.fragment.encode('ascii')
    return self