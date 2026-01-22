import collections
import functools
import inspect
import re
from tensorflow.python.framework import strict_mode
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls
def _add_deprecated_function_notice_to_docstring(doc, date, instructions):
    """Adds a deprecation notice to a docstring for deprecated functions."""
    main_text = ['THIS FUNCTION IS DEPRECATED. It will be removed %s.' % ('in a future version' if date is None else 'after %s' % date)]
    if instructions:
        main_text.append('Instructions for updating:')
    return decorator_utils.add_notice_to_docstring(doc, instructions, 'DEPRECATED FUNCTION', '(deprecated)', main_text, notice_type='Deprecated')