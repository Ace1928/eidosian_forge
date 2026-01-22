def _cell_html(self, printer, A, ri, ci=None):
    args = []
    if ci is not None and ri > ci:
        r = '-'
    else:
        if ci is None:
            c = A[ri]
            is_missing = self.missing[ri]
        else:
            c = A[ri][ci]
            is_missing = self.missing[ri] or self.missing[ci]
        if c is None:
            r = ''
        else:
            r = ', '.join((self._cell_label_html(printer, *r) for r in c))
        if is_missing:
            args.append('style="background-color: #%s;"' % self.missing_color)
    return '<td %s>%s</td>' % (' '.join(args), r)