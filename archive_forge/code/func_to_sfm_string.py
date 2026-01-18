import codecs
import re
from io import StringIO
from xml.etree.ElementTree import Element, ElementTree, SubElement, TreeBuilder
from nltk.data import PathPointer, find
def to_sfm_string(tree, encoding=None, errors='strict', unicode_fields=None):
    """
    Return a string with a standard format representation of the toolbox
    data in tree (tree can be a toolbox database or a single record).

    :param tree: flat representation of toolbox data (whole database or single record)
    :type tree: ElementTree._ElementInterface
    :param encoding: Name of an encoding to use.
    :type encoding: str
    :param errors: Error handling scheme for codec. Same as the ``encode()``
        builtin string method.
    :type errors: str
    :param unicode_fields:
    :type unicode_fields: dict(str) or set(str)
    :rtype: str
    """
    if tree.tag == 'record':
        root = Element('toolbox_data')
        root.append(tree)
        tree = root
    if tree.tag != 'toolbox_data':
        raise ValueError('not a toolbox_data element structure')
    if encoding is None and unicode_fields is not None:
        raise ValueError('if encoding is not specified then neither should unicode_fields')
    l = []
    for rec in tree:
        l.append('\n')
        for field in rec:
            mkr = field.tag
            value = field.text
            if encoding is not None:
                if unicode_fields is not None and mkr in unicode_fields:
                    cur_encoding = 'utf8'
                else:
                    cur_encoding = encoding
                if re.search(_is_value, value):
                    l.append(f'\\{mkr} {value}\n'.encode(cur_encoding, errors))
                else:
                    l.append(f'\\{mkr}{value}\n'.encode(cur_encoding, errors))
            elif re.search(_is_value, value):
                l.append(f'\\{mkr} {value}\n')
            else:
                l.append(f'\\{mkr}{value}\n')
    return ''.join(l[1:])