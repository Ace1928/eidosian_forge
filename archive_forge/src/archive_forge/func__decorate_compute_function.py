from pyarrow._compute import (  # noqa
from collections import namedtuple
import inspect
from textwrap import dedent
import warnings
import pyarrow as pa
from pyarrow import _compute_docstrings
from pyarrow.vendored import docscrape
def _decorate_compute_function(wrapper, exposed_name, func, options_class):
    cpp_doc = func._doc
    wrapper.__arrow_compute_function__ = dict(name=func.name, arity=func.arity, options_class=cpp_doc.options_class, options_required=cpp_doc.options_required)
    wrapper.__name__ = exposed_name
    wrapper.__qualname__ = exposed_name
    doc_pieces = []
    summary = cpp_doc.summary
    if not summary:
        arg_str = 'arguments' if func.arity > 1 else 'argument'
        summary = 'Call compute function {!r} with the given {}'.format(func.name, arg_str)
    doc_pieces.append(f'{summary}.\n\n')
    description = cpp_doc.description
    if description:
        doc_pieces.append(f'{description}\n\n')
    doc_addition = _compute_docstrings.function_doc_additions.get(func.name)
    doc_pieces.append(dedent('        Parameters\n        ----------\n        '))
    arg_names = _get_arg_names(func)
    for arg_name in arg_names:
        if func.kind in ('vector', 'scalar_aggregate'):
            arg_type = 'Array-like'
        else:
            arg_type = 'Array-like or scalar-like'
        doc_pieces.append(f'{arg_name} : {arg_type}\n')
        doc_pieces.append('    Argument to compute function.\n')
    if options_class is not None:
        options_class_doc = _scrape_options_class_doc(options_class)
        if options_class_doc:
            for p in options_class_doc.params:
                doc_pieces.append(f'{p.name} : {p.type}\n')
                for s in p.desc:
                    doc_pieces.append(f'    {s}\n')
        else:
            warnings.warn(f'Options class {options_class.__name__} does not have a docstring', RuntimeWarning)
            options_sig = inspect.signature(options_class)
            for p in options_sig.parameters.values():
                doc_pieces.append(dedent('                {0} : optional\n                    Parameter for {1} constructor. Either `options`\n                    or `{0}` can be passed, but not both at the same time.\n                '.format(p.name, options_class.__name__)))
        doc_pieces.append(dedent(f'            options : pyarrow.compute.{options_class.__name__}, optional\n                Alternative way of passing options.\n            '))
    doc_pieces.append(dedent('        memory_pool : pyarrow.MemoryPool, optional\n            If not passed, will allocate memory from the default memory pool.\n        '))
    if doc_addition is not None:
        doc_pieces.append('\n{}\n'.format(dedent(doc_addition).strip('\n')))
    wrapper.__doc__ = ''.join(doc_pieces)
    return wrapper