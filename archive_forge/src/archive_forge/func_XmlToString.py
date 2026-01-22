import sys
import re
import os
import locale
from functools import reduce
def XmlToString(content, encoding='utf-8', pretty=False):
    """ Writes the XML content to disk, touching the file only if it has changed.

  Visual Studio files have a lot of pre-defined structures.  This function makes
  it easy to represent these structures as Python data structures, instead of
  having to create a lot of function calls.

  Each XML element of the content is represented as a list composed of:
  1. The name of the element, a string,
  2. The attributes of the element, a dictionary (optional), and
  3+. The content of the element, if any.  Strings are simple text nodes and
      lists are child elements.

  Example 1:
      <test/>
  becomes
      ['test']

  Example 2:
      <myelement a='value1' b='value2'>
         <childtype>This is</childtype>
         <childtype>it!</childtype>
      </myelement>

  becomes
      ['myelement', {'a':'value1', 'b':'value2'},
         ['childtype', 'This is'],
         ['childtype', 'it!'],
      ]

  Args:
    content:  The structured content to be converted.
    encoding: The encoding to report on the first XML line.
    pretty: True if we want pretty printing with indents and new lines.

  Returns:
    The XML content as a string.
  """
    xml_parts = ['<?xml version="1.0" encoding="%s"?>' % encoding]
    if pretty:
        xml_parts.append('\n')
    _ConstructContentList(xml_parts, content, pretty)
    return ''.join(xml_parts)