import sys
import time
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ..core.options import Options, Store, StoreOptions, options_policy
from ..core.pprint import InfoPrinter
from ..operation import Compositor
from IPython.core import page
@magics_class
class OptsMagic(Magics):
    """
    Magic for easy customising of normalization, plot and style options.
    Consult %%opts? for more information.
    """
    error_message = None
    opts_spec = None
    strict = False

    @classmethod
    def process_element(cls, obj):
        """
        To be called by the display hook which supplies the element to
        be displayed. Any customisation of the object can then occur
        before final display. If there is any error, a HTML message
        may be returned. If None is returned, display will proceed as
        normal.
        """
        if cls.error_message:
            if cls.strict:
                return cls.error_message
            else:
                sys.stderr.write(cls.error_message)
        if cls.opts_spec is not None:
            StoreOptions.set_options(obj, cls.opts_spec)
            cls.opts_spec = None
        return None

    @classmethod
    def register_custom_spec(cls, spec):
        spec, _ = StoreOptions.expand_compositor_keys(spec)
        errmsg = StoreOptions.validation_error_message(spec)
        if errmsg:
            cls.error_message = errmsg
        cls.opts_spec = spec

    @classmethod
    def _partition_lines(cls, line, cell):
        """
        Check the code for additional use of %%opts. Enables
        multi-line use of %%opts in a single call to the magic.
        """
        if cell is None:
            return (line, cell)
        specs, code = ([line], [])
        for line in cell.splitlines():
            if line.strip().startswith('%%opts'):
                specs.append(line.strip()[7:])
            else:
                code.append(line)
        return (' '.join(specs), '\n'.join(code))

    @line_cell_magic
    def opts(self, line='', cell=None):
        """
        The opts line/cell magic with tab-completion.

        %%opts [ [path] [normalization] [plotting options] [style options]]+

        path:             A dotted type.group.label specification
                          (e.g. Image.Grayscale.Photo)

        normalization:    List of normalization options delimited by braces.
                          One of | -axiswise | -framewise | +axiswise | +framewise |
                          E.g. { +axiswise +framewise }

        plotting options: List of plotting option keywords delimited by
                          square brackets. E.g. [show_title=False]

        style options:    List of style option keywords delimited by
                          parentheses. E.g. (lw=10 marker='+')

        Note that commas between keywords are optional (not
        recommended) and that keywords must end in '=' without a
        separating space.

        More information may be found in the class docstring of
        util.parser.OptsSpec.
        """
        line, cell = self._partition_lines(line, cell)
        try:
            spec = OptsSpec.parse(line, ns=self.shell.user_ns)
        except SyntaxError:
            display(HTML('<b>Invalid syntax</b>: Consult <tt>%%opts?</tt> for more information.'))
            return
        available_elements = set()
        for backend in Store.loaded_backends():
            available_elements |= set(Store.options(backend).children)
        spec_elements = {k.split('.')[0] for k in spec.keys()}
        unknown_elements = spec_elements - available_elements
        if unknown_elements:
            msg = '<b>WARNING:</b> Unknown elements {unknown} not registered with any of the loaded backends.'
            display(HTML(msg.format(unknown=', '.join(unknown_elements))))
        if cell:
            self.register_custom_spec(spec)
            self.shell.run_cell(cell, store_history=STORE_HISTORY)
        else:
            errmsg = StoreOptions.validation_error_message(spec)
            if errmsg:
                OptsMagic.error_message = None
                sys.stderr.write(errmsg)
                if self.strict:
                    display(HTML('Options specification will not be applied.'))
                    return
            with options_policy(skip_invalid=True, warn_on_skip=False):
                StoreOptions.apply_customizations(spec, Store.options())
        OptsMagic.error_message = None