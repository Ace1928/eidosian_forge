import inspect
import textwrap
import torch.jit
from torch.jit._builtins import _find_builtin
def _get_global_builtins():
    supported_builtins = ['print', 'tuple', 'float', 'complex', 'int', 'bool', 'str', 'getattr', 'hasattr', 'isinstance', 'len', 'hex', 'oct', 'round', 'hash', 'min', 'max', 'abs', 'all', 'divmod', 'list', 'ord', 'chr', 'bin', 'range', 'zip', 'enumerate', 'sorted']
    op_renames = {'bool': 'aten::Bool', 'int': 'aten::Int', 'float': 'aten::Float', 'complex': 'aten::Complex', 'abs': 'prim::abs', 'max': 'prim::max', 'min': 'prim::min', 'range': 'fake::does_not_exist'}
    schemaless_op_explanations = {'print': 'Print any value', 'tuple': 'Lists cannot be converted to tuples with this method since their size is not statically known', 'getattr': 'Attribute name must be a literal string', 'hasattr': 'Attribute name must be a literal string', 'isinstance': 'Result is static', 'zip': 'Arguments must be iterable. See :ref:`Iterables <jit_iterables>` for details.', 'enumerate': 'Arguments must be iterable. See :ref:`Iterables <jit_iterables>` for details.', 'range': 'Can only be used as an iterator in a for loop'}
    magic_methods = [('complex', '__complex__'), ('float', '__float__'), ('int', '__int__'), ('bool', '__bool__'), ('str', '__str__'), ('len', '__len__'), ('hex', '__hex__'), ('oct', '__oct__')]
    magic_methods_rows = []
    for fn, magic_method in magic_methods:
        magic_methods_rows.append(f'"{fn}", "``{magic_method}``"')
    schematized_ops = []
    schemaless_ops = []
    for fn in supported_builtins:
        op_name = f'aten::{fn}'
        if fn in op_renames:
            op_name = op_renames[fn]
        schemas = torch._C._jit_get_schemas_for_operator(op_name)
        for s in schemas:
            schematized_ops.append(_emit_schema(None, fn, s, padding=0))
        if len(schemas) > 0:
            schematized_ops.append('')
        else:
            table_row = f'":any:`{fn}`", "{schemaless_op_explanations[fn]}"'
            schemaless_ops.append(table_row)
    schematized_ops_str = '\n'.join(schematized_ops)
    schemaless_ops_str = '\n'.join(schemaless_ops)
    magic_methods_rows_str = '\n'.join(magic_methods_rows)
    schematized_ops_str = textwrap.indent(schematized_ops_str, '\t')
    schemaless_ops_str = textwrap.indent(schemaless_ops_str, '\t')
    magic_methods_rows_str = textwrap.indent(magic_methods_rows_str, '\t')
    section = f'\nThe functions in the following table are supported but do not have a static schema\n\n.. csv-table::\n    :header: "Function", "Note"\n\n{schemaless_ops_str}\n\nThe following functions will use the corresponding magic method on :any:`TorchScript classes`\n\n.. csv-table::\n    :header: "Function", "Magic Method"\n\n{magic_methods_rows_str}\n\nThese built-in functions use the schema\n\n.. rst-class:: codeblock-height-limiter\n\n::\n\n{schematized_ops_str}\n    '
    return ('Python Built-in Functions', section)