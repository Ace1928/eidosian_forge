import codecs
from xml.sax.saxutils import escape, quoteattr
Compute qname for a uri using our extra namespaces,
        or the given namespace manager