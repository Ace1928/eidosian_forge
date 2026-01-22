import json
from .web import CSSPrinter, _html_clsname
def _print_ReactionSystem(self, rsys, **kwargs):
    tab = super(JSPrinter, self)._print_ReactionSystem(rsys, **kwargs)
    _script_tag = '<script type="text/javascript">%s</script>'
    substances = self._get('substances', **kwargs)
    cls_names_substances = list(map(_html_clsname, substances))
    return tab + _script_tag % _js_rsys(cls_names_substances=cls_names_substances, substance_row_cls_irrel={cns: [self._tr_id(rsys, i) for i in range(rsys.nr) if i not in rsys.substance_participation(sk)] for cns, sk in zip(cls_names_substances, substances)}, rsys_id=id(rsys))