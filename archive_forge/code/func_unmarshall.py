from __future__ import unicode_literals
import sys
import logging
import re
import time
import xml.dom.minidom
from . import __author__, __copyright__, __license__, __version__
from .helpers import TYPE_MAP, TYPE_MARSHAL_FN, TYPE_UNMARSHAL_FN, \
def unmarshall(self, types, strict=True):
    """Convert to python values the current serialized xml element"""
    d = {}
    for node in self():
        name = str(node.get_local_name())
        ref_name_type = None
        if 'href' in node.attributes().keys():
            href = node['href'][1:]
            for ref_node in self(root=True)('multiRef'):
                if ref_node['id'] == href:
                    node = ref_node
                    ref_name_type = ref_node['xsi:type'].split(':')[1]
                    break
        try:
            if isinstance(types, dict):
                fn = types[name]
                if any([k for k, v in node[:] if 'arrayType' in k]) and (not isinstance(fn, list)):
                    fn = [fn]
            else:
                fn = types
        except (KeyError,) as e:
            xmlns = node['xmlns'] or node.get_namespace_uri(node.get_prefix())
            if 'xsi:type' in node.attributes().keys():
                xsd_type = node['xsi:type'].split(':')[1]
                try:
                    if xsd_type == 'Array':
                        array_type = [k for k, v in node[:] if 'arrayType' in k][0]
                        xsd_type = node[array_type].split(':')[1]
                        if '[' in xsd_type:
                            xsd_type = xsd_type[:xsd_type.index('[')]
                        fn = [REVERSE_TYPE_MAP[xsd_type]]
                    else:
                        fn = REVERSE_TYPE_MAP[xsd_type]
                except:
                    fn = None
            elif xmlns == 'http://www.w3.org/2001/XMLSchema':
                fn = None
            elif None in types:
                fn = None
            elif strict:
                raise TypeError('Tag: %s invalid (type not found)' % (name,))
            else:
                fn = str
        if isinstance(fn, list):
            value = d.setdefault(name, [])
            children = node.children() or node
            if fn and (not isinstance(fn[0], dict)):
                for child in children or []:
                    tmp_dict = child.unmarshall(fn[0], strict)
                    value.extend(tmp_dict.values())
            elif len(fn[0]) > 1:
                for parent in node:
                    tmp_dict = {}
                    for child in node.children() or []:
                        tmp_dict.update(child.unmarshall(fn[0], strict))
                    value.append(tmp_dict)
            else:
                for child in children or []:
                    value.append(child.unmarshall(fn[0], strict))
        elif isinstance(fn, tuple):
            value = []
            _d = {}
            children = node.children()
            as_dict = len(fn) == 1 and isinstance(fn[0], dict)
            for child in children and children() or []:
                if as_dict:
                    _d.update(child.unmarshall(fn[0], strict))
                else:
                    value.append(child.unmarshall(fn[0], strict))
            if as_dict:
                value.append(_d)
            if name in d:
                _tmp = list(d[name])
                _tmp.extend(value)
                value = tuple(_tmp)
            else:
                value = tuple(value)
        elif isinstance(fn, dict):
            children = node.children()
            value = children and children.unmarshall(fn, strict)
        elif fn is None:
            value = node
        elif unicode(node) or (fn == str and unicode(node) != ''):
            try:
                fn = TYPE_UNMARSHAL_FN.get(fn, fn)
                if fn == str:
                    value = unicode(node)
                else:
                    value = fn(unicode(node))
            except (ValueError, TypeError) as e:
                raise ValueError('Tag: %s: %s' % (name, e))
        else:
            value = None
        d[name] = value
    return d