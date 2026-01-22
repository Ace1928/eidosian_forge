from sympy.external.importtools import version_tuple
from io import BytesIO
from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable
def _init_ipython_printing(ip, stringify_func, use_latex, euler, forecolor, backcolor, fontsize, latex_mode, print_builtin, latex_printer, scale, **settings):
    """Setup printing in IPython interactive session. """
    try:
        from IPython.lib.latextools import latex_to_png
    except ImportError:
        pass
    if forecolor is None:
        color = ip.colors.lower()
        if color == 'lightbg':
            forecolor = 'Black'
        elif color == 'linux':
            forecolor = 'White'
        else:
            forecolor = 'Gray'
        debug('init_printing: Automatic foreground color:', forecolor)
    if use_latex == 'svg':
        extra_preamble = '\n\\special{color %s}' % forecolor
    else:
        extra_preamble = ''
    imagesize = 'tight'
    offset = '0cm,0cm'
    resolution = round(150 * scale)
    dvi = '-T %s -D %d -bg %s -fg %s -O %s' % (imagesize, resolution, backcolor, forecolor, offset)
    dvioptions = dvi.split()
    svg_scale = 150 / 72 * scale
    dvioptions_svg = ['--no-fonts', '--scale={}'.format(svg_scale)]
    debug('init_printing: DVIOPTIONS:', dvioptions)
    debug('init_printing: DVIOPTIONS_SVG:', dvioptions_svg)
    latex = latex_printer or default_latex

    def _print_plain(arg, p, cycle):
        """caller for pretty, for use in IPython 0.11"""
        if _can_print(arg):
            p.text(stringify_func(arg))
        else:
            p.text(IPython.lib.pretty.pretty(arg))

    def _preview_wrapper(o):
        exprbuffer = BytesIO()
        try:
            preview(o, output='png', viewer='BytesIO', euler=euler, outputbuffer=exprbuffer, extra_preamble=extra_preamble, dvioptions=dvioptions, fontsize=fontsize)
        except Exception as e:
            debug('png printing:', '_preview_wrapper exception raised:', repr(e))
            raise
        return exprbuffer.getvalue()

    def _svg_wrapper(o):
        exprbuffer = BytesIO()
        try:
            preview(o, output='svg', viewer='BytesIO', euler=euler, outputbuffer=exprbuffer, extra_preamble=extra_preamble, dvioptions=dvioptions_svg, fontsize=fontsize)
        except Exception as e:
            debug('svg printing:', '_preview_wrapper exception raised:', repr(e))
            raise
        return exprbuffer.getvalue().decode('utf-8')

    def _matplotlib_wrapper(o):
        try:
            try:
                return latex_to_png(o, color=forecolor, scale=scale)
            except TypeError:
                return latex_to_png(o)
        except ValueError as e:
            debug('matplotlib exception caught:', repr(e))
            return None
    printing_hooks = ('_latex', '_sympystr', '_pretty', '_sympyrepr')

    def _can_print(o):
        """Return True if type o can be printed with one of the SymPy printers.

        If o is a container type, this is True if and only if every element of
        o can be printed in this way.
        """
        try:
            builtin_types = (list, tuple, set, frozenset)
            if isinstance(o, builtin_types):
                if type(o).__str__ not in (i.__str__ for i in builtin_types) or type(o).__repr__ not in (i.__repr__ for i in builtin_types):
                    return False
                return all((_can_print(i) for i in o))
            elif isinstance(o, dict):
                return all((_can_print(i) and _can_print(o[i]) for i in o))
            elif isinstance(o, bool):
                return False
            elif isinstance(o, Printable):
                return True
            elif any((hasattr(o, hook) for hook in printing_hooks)):
                return True
            elif isinstance(o, (float, int)) and print_builtin:
                return True
            return False
        except RuntimeError:
            return False

    def _print_latex_png(o):
        """
        A function that returns a png rendered by an external latex
        distribution, falling back to matplotlib rendering
        """
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                s = '$\\displaystyle %s$' % s
            try:
                return _preview_wrapper(s)
            except RuntimeError as e:
                debug('preview failed with:', repr(e), ' Falling back to matplotlib backend')
                if latex_mode != 'inline':
                    s = latex(o, mode='inline', **settings)
                return _matplotlib_wrapper(s)

    def _print_latex_svg(o):
        """
        A function that returns a svg rendered by an external latex
        distribution, no fallback available.
        """
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                s = '$\\displaystyle %s$' % s
            try:
                return _svg_wrapper(s)
            except RuntimeError as e:
                debug('preview failed with:', repr(e), ' No fallback available.')

    def _print_latex_matplotlib(o):
        """
        A function that returns a png rendered by mathtext
        """
        if _can_print(o):
            s = latex(o, mode='inline', **settings)
            return _matplotlib_wrapper(s)

    def _print_latex_text(o):
        """
        A function to generate the latex representation of SymPy expressions.
        """
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                return '$\\displaystyle %s$' % s
            return s

    def _result_display(self, arg):
        """IPython's pretty-printer display hook, for use in IPython 0.10

           This function was adapted from:

            ipython/IPython/hooks.py:155

        """
        if self.rc.pprint:
            out = stringify_func(arg)
            if '\n' in out:
                print()
            print(out)
        else:
            print(repr(arg))
    import IPython
    if version_tuple(IPython.__version__) >= version_tuple('0.11'):
        printable_types = [float, tuple, list, set, frozenset, dict, int]
        plaintext_formatter = ip.display_formatter.formatters['text/plain']
        for cls in printable_types + [Printable]:
            plaintext_formatter.for_type(cls, _print_plain)
        svg_formatter = ip.display_formatter.formatters['image/svg+xml']
        if use_latex in ('svg',):
            debug('init_printing: using svg formatter')
            for cls in printable_types:
                svg_formatter.for_type(cls, _print_latex_svg)
            Printable._repr_svg_ = _print_latex_svg
        else:
            debug('init_printing: not using any svg formatter')
            for cls in printable_types:
                if cls in svg_formatter.type_printers:
                    svg_formatter.type_printers.pop(cls)
            Printable._repr_svg_ = Printable._repr_disabled
        png_formatter = ip.display_formatter.formatters['image/png']
        if use_latex in (True, 'png'):
            debug('init_printing: using png formatter')
            for cls in printable_types:
                png_formatter.for_type(cls, _print_latex_png)
            Printable._repr_png_ = _print_latex_png
        elif use_latex == 'matplotlib':
            debug('init_printing: using matplotlib formatter')
            for cls in printable_types:
                png_formatter.for_type(cls, _print_latex_matplotlib)
            Printable._repr_png_ = _print_latex_matplotlib
        else:
            debug('init_printing: not using any png formatter')
            for cls in printable_types:
                if cls in png_formatter.type_printers:
                    png_formatter.type_printers.pop(cls)
            Printable._repr_png_ = Printable._repr_disabled
        latex_formatter = ip.display_formatter.formatters['text/latex']
        if use_latex in (True, 'mathjax'):
            debug('init_printing: using mathjax formatter')
            for cls in printable_types:
                latex_formatter.for_type(cls, _print_latex_text)
            Printable._repr_latex_ = _print_latex_text
        else:
            debug('init_printing: not using text/latex formatter')
            for cls in printable_types:
                if cls in latex_formatter.type_printers:
                    latex_formatter.type_printers.pop(cls)
            Printable._repr_latex_ = Printable._repr_disabled
    else:
        ip.set_hook('result_display', _result_display)