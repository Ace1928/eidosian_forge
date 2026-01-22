import inspect
import numbers
import os
import re
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
def _add_np_doc(doc, np_fun_name, np_f, link):
    """Appends the numpy docstring to `doc`, according to `set_np_doc_form`.

  See `set_np_doc_form` for how it controls the form of the numpy docstring.

  Args:
    doc: the docstring to be appended to.
    np_fun_name: the name of the numpy function.
    np_f: (optional) the numpy function.
    link: (optional) which link to use. See `np_doc` for details.

  Returns:
    `doc` with numpy docstring appended.
  """
    flag = get_np_doc_form()
    if flag == 'inlined':
        if _has_docstring(np_f):
            doc += 'Documentation for `numpy.%s`:\n\n' % np_fun_name
            doc += np_f.__doc__.replace('>>>', '>')
    elif isinstance(flag, str):
        if link is None:
            url = generate_link(flag, np_fun_name)
        elif isinstance(link, AliasOf):
            url = generate_link(flag, link.value)
        elif isinstance(link, Link):
            url = link.value
        else:
            url = None
        if url is not None:
            if is_check_link():
                import requests
                r = requests.head(url)
                if r.status_code != 200:
                    raise ValueError(f'Check link failed at [{url}] with status code {r.status_code}. Argument `np_fun_name` is {np_fun_name}.')
            doc += 'See the NumPy documentation for [`numpy.%s`](%s).' % (np_fun_name, url)
    return doc