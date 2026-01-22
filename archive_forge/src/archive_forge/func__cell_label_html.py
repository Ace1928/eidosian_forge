def _cell_label_html(self, printer, ori_idx, rxn):
    """Reaction formatting callback. (reaction index -> string)"""
    pretty = rxn.unicode(self.substances, with_param=True, with_name=False)
    return '<a title="%d: %s">%s</a>' % (ori_idx, pretty, printer._print(rxn.name or rxn.param))