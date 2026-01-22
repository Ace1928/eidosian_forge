import sys
import time
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ..core.options import Options, Store, StoreOptions, options_policy
from ..core.pprint import InfoPrinter
from ..operation import Compositor
from IPython.core import page
@magics_class
class CompositorMagic(Magics):
    """
    Magic allowing easy definition of compositor operations.
    Consult %compositor? for more information.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lines = ['The %compositor line magic is used to define compositors.']
        self.compositor.__func__.__doc__ = '\n'.join(lines + [CompositorSpec.__doc__])

    @line_magic
    def compositor(self, line):
        if line.strip():
            for definition in CompositorSpec.parse(line.strip(), ns=self.shell.user_ns):
                group = {group: Options() for group in Options._option_groups}
                type_name = definition.output_type.__name__
                Store.options()[type_name + '.' + definition.group] = group
                Compositor.register(definition)
        else:
            print('For help with the %compositor magic, call %compositor?\n')

    @classmethod
    def option_completer(cls, k, v):
        line = v.text_until_cursor
        operation_openers = [op.__name__ + '(' for op in Compositor.operations]
        modes = ['data', 'display']
        op_declared = any((op in line for op in operation_openers))
        mode_declared = any((mode in line for mode in modes))
        if not mode_declared:
            return modes
        elif not op_declared:
            return operation_openers
        if op_declared and ')' not in line:
            return [')']
        elif line.split(')')[1].strip() and '[' not in line:
            return ['[']
        elif '[' in line:
            return [']']