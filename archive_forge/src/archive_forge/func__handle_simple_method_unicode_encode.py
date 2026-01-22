from __future__ import absolute_import
import re
import sys
import copy
import codecs
import itertools
from . import TypeSlots
from .ExprNodes import not_a_constant
import cython
from . import Nodes
from . import ExprNodes
from . import PyrexTypes
from . import Visitor
from . import Builtin
from . import UtilNodes
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .StringEncoding import EncodedString, bytes_literal, encoded_string
from .Errors import error, warning
from .ParseTreeTransforms import SkipDeclarations
from .. import Utils
def _handle_simple_method_unicode_encode(self, node, function, args, is_unbound_method):
    """Replace unicode.encode(...) by a direct C-API call to the
        corresponding codec.
        """
    if len(args) < 1 or len(args) > 3:
        self._error_wrong_arg_count('unicode.encode', node, args, '1-3')
        return node
    string_node = args[0]
    if len(args) == 1:
        null_node = ExprNodes.NullNode(node.pos)
        return self._substitute_method_call(node, function, 'PyUnicode_AsEncodedString', self.PyUnicode_AsEncodedString_func_type, 'encode', is_unbound_method, [string_node, null_node, null_node])
    parameters = self._unpack_encoding_and_error_mode(node.pos, args)
    if parameters is None:
        return node
    encoding, encoding_node, error_handling, error_handling_node = parameters
    if encoding and isinstance(string_node, ExprNodes.UnicodeNode):
        try:
            value = string_node.value.encode(encoding, error_handling)
        except:
            pass
        else:
            value = bytes_literal(value, encoding)
            return ExprNodes.BytesNode(string_node.pos, value=value, type=Builtin.bytes_type)
    if encoding and error_handling == 'strict':
        codec_name = self._find_special_codec_name(encoding)
        if codec_name is not None and '-' not in codec_name:
            encode_function = 'PyUnicode_As%sString' % codec_name
            return self._substitute_method_call(node, function, encode_function, self.PyUnicode_AsXyzString_func_type, 'encode', is_unbound_method, [string_node])
    return self._substitute_method_call(node, function, 'PyUnicode_AsEncodedString', self.PyUnicode_AsEncodedString_func_type, 'encode', is_unbound_method, [string_node, encoding_node, error_handling_node])