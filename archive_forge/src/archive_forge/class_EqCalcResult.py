from itertools import product
import math
from .printing import number_to_scientific_html
from ._util import get_backend, mat_dot_vec, prodpow
class EqCalcResult(object):
    attrs = {'sane': bool, 'success': bool, 'nfev': int, 'njev': int, 'time_cpu': float, 'time_wall': float}

    def __init__(self, eqsys, init_concs, varied):
        self.eqsys = eqsys
        self.all_inits, self.varied_keys = self.eqsys.per_substance_varied(init_concs, varied)
        self.conc = np.empty_like(self.all_inits)
        for k, v in self.attrs.items():
            setattr(self, k, np.zeros(self.all_inits.shape[:-1], dtype=v))

    def solve(self, **kwargs):
        for index in product(*map(range, self.all_inits.shape[:-1])):
            slc = tuple(index) + (slice(None),)
            self.conc[slc], nfo, sane = self.eqsys._solve(self.all_inits[slc], **kwargs)
            self.sane[index] = sane

            def _get(k):
                try:
                    return nfo[k]
                except TypeError:
                    return nfo[-1][k]
            for k in self.attrs:
                if k == 'sane':
                    continue
                try:
                    getattr(self, k)[index] = _get(k)
                except KeyError:
                    pass

    def _repr_html_(self):

        def fmt(num):
            return number_to_scientific_html(num, fmt=5)
        if len(self.varied_keys) == 0:
            raise NotImplementedError()
        elif len(self.varied_keys) == 1:
            var_html = self.eqsys.substances[self.varied_keys[0]].html_name
            header = ['[%s]<sub>0</sub>' % var_html] + ['[%s]' % s.html_name for s in self.eqsys.substances.values()]

            def row(i):
                j = self.eqsys.as_substance_index(self.varied_keys[0])
                return map(fmt, [self.all_inits[i, j]] + self.conc[i, :].tolist())
            pre = "  <td style='font-weight: bold;'>\n      "
            linker = '\n    </td>\n    <td>\n      '
            post = '\n    </td>'
            rows = [pre + linker.join(row(i)) + post for i in range(self.all_inits.shape[0])]
            template = '<table>\n  <tr>\n    <th>\n    %s\n    </th>\n  </tr>\n  <tr>\n  %s\n  </tr>\n</table>'
            head_linker = '\n    </th>\n    <th>\n      '
            row_linker = '\n  </tr>\n  <tr>\n  '
            return template % (head_linker.join(header), row_linker.join(rows))
        else:
            raise NotImplementedError()

    def plot(self, ls=('-', '--', ':', '-.'), c=('k', 'r', 'g', 'b', 'c', 'm', 'y'), latex=None):
        import matplotlib.pyplot as plt
        if latex is None:
            latex = next(iter(self.eqsys.substances.values())).latex_name is not None
        if len(self.varied_keys) == 0:
            raise NotImplementedError()
        elif len(self.varied_keys) == 1:
            x = self.all_inits[:, self.eqsys.as_substance_index(self.varied_keys[0])]
            for idx, (k, v) in enumerate(self.eqsys.substances.items()):
                lbl = '$\\mathrm{' + v.latex_name + '}$' if latex else v.name
                plt.plot(x, self.conc[:, idx], label=lbl, ls=ls[idx % len(ls)], c=c[idx % len(c)])
            ax = plt.gca()
            ax.set_xscale('log')
            ax.set_yscale('log')
            var_latex = self.eqsys.substances[self.varied_keys[0]].latex_name
            ax.set_xlabel(('$[\\mathrm{%s}]_0$' if latex else '[%s]0') % var_latex)
            ax.set_ylabel('Concentration')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        else:
            raise NotImplementedError()