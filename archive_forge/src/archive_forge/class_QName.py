from functools import reduce
import sys
from itertools import chain
import operator
import six
from genshi.compat import stringrepr
from genshi.util import stripentities, striptags
class QName(six.text_type):
    """A qualified element or attribute name.
    
    The unicode value of instances of this class contains the qualified name of
    the element or attribute, in the form ``{namespace-uri}local-name``. The
    namespace URI can be obtained through the additional `namespace` attribute,
    while the local name can be accessed through the `localname` attribute.
    
    >>> qname = QName('foo')
    >>> qname
    QName('foo')
    >>> qname.localname
    'foo'
    >>> qname.namespace
    
    >>> qname = QName('http://www.w3.org/1999/xhtml}body')
    >>> qname
    QName('http://www.w3.org/1999/xhtml}body')
    >>> qname.localname
    'body'
    >>> qname.namespace
    'http://www.w3.org/1999/xhtml'
    """
    __slots__ = ['namespace', 'localname']

    def __new__(cls, qname):
        """Create the `QName` instance.
        
        :param qname: the qualified name as a string of the form
                      ``{namespace-uri}local-name``, where the leading curly
                      brace is optional
        """
        if type(qname) is cls:
            return qname
        qname = qname.lstrip('{')
        parts = qname.split('}', 1)
        if len(parts) > 1:
            self = six.text_type.__new__(cls, '{%s' % qname)
            self.namespace, self.localname = map(six.text_type, parts)
        else:
            self = six.text_type.__new__(cls, qname)
            self.namespace, self.localname = (None, six.text_type(qname))
        return self

    def __getnewargs__(self):
        return (self.lstrip('{'),)
    if sys.version_info[0] == 2:

        def __repr__(self):
            return '%s(%s)' % (type(self).__name__, stringrepr(self.lstrip('{')))
    else:

        def __repr__(self):
            return '%s(%r)' % (type(self).__name__, self.lstrip('{'))