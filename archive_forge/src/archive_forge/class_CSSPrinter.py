from .numbers import number_to_scientific_html
from .string import StrPrinter
class CSSPrinter(HTMLPrinter):

    def _print_Substance(self, s, **kwargs):
        key = s.name
        name = s.html_name or key
        common_sty = 'border-radius: 5pt; padding: 0pt 3pt 0pt 3pt;'
        colors = self._get('colors', **kwargs)
        if key in colors:
            style = 'background-color:#%s; border: 1px solid #%s; %s' % (colors[key] + (common_sty,))
        else:
            style = common_sty
        fmt = '<span class="%s" style="%s">%s</span>'
        return fmt % (_html_clsname(key), style, name)

    def _tr_id(self, rsys, i):
        return 'chempy_%d_%d' % (id(rsys), i)

    def _print_ReactionSystem(self, rsys, **kwargs):
        sep = '</td><td style="text-align:left;">&nbsp;'
        around = ('</td><td style="text-align:center;">', '</td><td style="text-align:left;">')
        row_template = '<tr class="%s"><td style="text-align:right;">%s</td></tr>'
        rows = [row_template % (self._tr_id(rsys, i), s) for i, s in enumerate(map(lambda r: self._print(r, Reaction_param_separator=sep, Reaction_around_arrow=around), rsys.rxns))]
        tab_template = '<table class="chempy_ReactionSystem chempy_%d">%s%s</table>'
        header = '<tr><th style="text-align:center;" colspan="5">%s</th></tr>' % (rsys.name or '')
        return tab_template % (id(rsys), header, '\n\n'.join(rows))