import io
import sys
import six
import types
from six import StringIO
from io import BytesIO
from lxml import etree
from ncclient import NCClientError
def yang_action(name, attrs):
    """Instantiate a YANG action element

    Args:
        name: A string representing the first descendant name of the
            XML element for the YANG action.
        attrs: A dict of attributes to apply to the XML element
            (e.g. namespaces).
    Returns:
        A tuple of 'lxml.etree._Element' values.  The first value
        represents the top-level YANG action element and the second
        represents the caller supplied initial node.
    """
    node = new_ele('action', attrs={'xmlns': YANG_NS_1_0})
    return (node, sub_ele(node, name, attrs))