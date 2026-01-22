class BimolecularTable(_RxnTable):
    """Table of bimolecular reactions

    Parameters
    ----------
    idx_rxn_pairs : iterable of (int, Reaction) pairs
    substances : dict
        Mapping substance key to Substance instance.
    sinks_sources_disjoint : tuple, None or True
        Colors sinks & sources. When ``True`` :meth:`sinks_sources_disjoint` is called.

    Returns
    -------
    string: html representation
    list: reactions not considered
    """
    _rsys_meth = '_bimolecular_reactions'

    def _html(self, printer, **kwargs):
        if 'substances' not in kwargs:
            kwargs['substances'] = self.substances
        ss = printer._get('substances', **kwargs)
        header = '<th></th>' + ''.join(('<th>%s</th>' % printer._print(s) for s in ss.values()))
        rows = ['<tr><td>%s</td>%s</tr>' % (printer._print(s), ''.join((self._cell_html(printer, self.idx_rxn_pairs, rowi, ci) for ci in range(len(ss))))) for rowi, s in enumerate(ss.values())]
        return '<table>%s</table>' % '\n'.join([header, '\n'.join(rows)])