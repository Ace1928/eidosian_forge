from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def SymmetricMapping(subcon, mapping, default=NotImplemented):
    """defines a symmetrical mapping: a->b, b->a.
    * subcon - the subcon to map
    * mapping - the encoding mapping (a dict); the decoding mapping is
      achieved by reversing this mapping
    * default - the default value to use when no mapping is found. if no
      default value is given, and exception is raised. setting to Pass would
      return the value "as is" (unmapped)
    """
    reversed_mapping = dict(((v, k) for k, v in mapping.items()))
    return MappingAdapter(subcon, encoding=mapping, decoding=reversed_mapping, encdefault=default, decdefault=default)