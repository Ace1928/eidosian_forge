import functools
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Dict, List
from . import language as tl
from ._C.libtriton.triton import runtime
class Mark:

    def __init__(self, fn, benchmarks):
        self.fn = fn
        self.benchmarks = benchmarks

    def _run(self, bench: Benchmark, save_path: str, show_plots: bool, print_data: bool, diff_col=False, **kwrags):
        import os
        import matplotlib.pyplot as plt
        import pandas as pd
        y_mean = bench.line_names
        y_min = [f'{x}-min' for x in bench.line_names]
        y_max = [f'{x}-max' for x in bench.line_names]
        x_names = list(bench.x_names)
        df = pd.DataFrame(columns=x_names + y_mean + y_min + y_max)
        for x in bench.x_vals:
            if not isinstance(x, (list, tuple)):
                x = [x for _ in x_names]
            if len(x) != len(x_names):
                raise ValueError(f'Expected {len(x_names)} values, got {x}')
            x_args = dict(zip(x_names, x))
            row_mean, row_min, row_max = ([], [], [])
            for y in bench.line_vals:
                ret = self.fn(**x_args, **{bench.line_arg: y}, **bench.args, **kwrags)
                try:
                    y_mean, y_min, y_max = ret
                except TypeError:
                    y_mean, y_min, y_max = (ret, None, None)
                row_mean += [y_mean]
                row_min += [y_min]
                row_max += [y_max]
            df.loc[len(df)] = list(x) + row_mean + row_min + row_max
        if bench.plot_name:
            plt.figure()
            ax = plt.subplot()
            first_x = x_names[0]
            for i, y in enumerate(bench.line_names):
                y_min, y_max = (df[y + '-min'], df[y + '-max'])
                col = bench.styles[i][0] if bench.styles else None
                sty = bench.styles[i][1] if bench.styles else None
                ax.plot(df[first_x], df[y], label=y, color=col, ls=sty)
                if not y_min.isnull().all() and (not y_max.isnull().all()):
                    y_min = y_min.astype(float)
                    y_max = y_max.astype(float)
                    ax.fill_between(df[first_x], y_min, y_max, alpha=0.15, color=col)
            ax.legend()
            ax.set_xlabel(bench.xlabel or first_x)
            ax.set_ylabel(bench.ylabel)
            ax.set_xscale('log' if bench.x_log else 'linear')
            ax.set_yscale('log' if bench.y_log else 'linear')
            if show_plots:
                plt.show()
            if save_path:
                plt.savefig(os.path.join(save_path, f'{bench.plot_name}.png'))
        df = df[x_names + bench.line_names]
        if diff_col and df.shape[1] == 2:
            col0, col1 = df.columns.tolist()
            df['Diff'] = df[col1] - df[col0]
        if print_data:
            print(bench.plot_name + ':')
            print(df)
        if save_path:
            df.to_csv(os.path.join(save_path, f'{bench.plot_name}.csv'), float_format='%.1f', index=False)
        return df

    def run(self, show_plots=False, print_data=False, save_path='', return_df=False, **kwargs):
        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks
        result_dfs = []
        if save_path:
            html = open(os.path.join(save_path, 'results.html'), 'w')
            html.write('<html><body>\n')
        for bench in benchmarks:
            result_dfs.append(self._run(bench, save_path, show_plots, print_data, **kwargs))
            if save_path:
                html.write(f'<image src="{bench.plot_name}.png"/>\n')
        if save_path:
            html.write('</body></html>\n')
        if return_df:
            if has_single_bench:
                return result_dfs[0]
            else:
                return result_dfs
        return None