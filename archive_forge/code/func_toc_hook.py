from ._base import DirectivePlugin
from ..toc import normalize_toc_item, render_toc_ul
def toc_hook(self, md, state):
    sections = []
    headings = []
    for tok in state.tokens:
        if tok['type'] == 'toc':
            sections.append(tok)
        elif tok['type'] == 'heading':
            headings.append(tok)
    if sections:
        toc_items = []
        for i, tok in enumerate(headings):
            tok['attrs']['id'] = self.generate_heading_id(tok, i)
            toc_items.append(normalize_toc_item(md, tok))
        for sec in sections:
            _min = sec['attrs']['min_level']
            _max = sec['attrs']['max_level']
            toc = [item for item in toc_items if _min <= item[0] <= _max]
            sec['attrs']['toc'] = toc