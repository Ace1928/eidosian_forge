from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.etree import ElementTree, SubElement
from lxml.etree import Comment, ProcessingInstruction
def _build_qname(self, ns_uri, local_name, nsmap, preferred_prefix, is_attribute):
    if ns_uri is None:
        return local_name
    if not is_attribute and nsmap.get(preferred_prefix) == ns_uri:
        prefix = preferred_prefix
    else:
        candidates = [pfx for pfx, uri in nsmap.items() if pfx is not None and uri == ns_uri]
        prefix = candidates[0] if len(candidates) == 1 else min(candidates) if candidates else None
    if prefix is None:
        return local_name
    return prefix + ':' + local_name