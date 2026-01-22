import os
import typing
def _import_cy_extensions():
    from ..cyextension import collections
    from ..cyextension import immutabledict
    from ..cyextension import processors
    from ..cyextension import resultproxy
    from ..cyextension import util
    return (collections, immutabledict, processors, resultproxy, util)