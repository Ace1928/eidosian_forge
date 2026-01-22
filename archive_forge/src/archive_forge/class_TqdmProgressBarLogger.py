from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import time
class TqdmProgressBarLogger(ProgressBarLogger):
    """Tqdm-powered progress bar for console or Notebooks.

    Parameters
    ----------
    init_state
      Initial state of the logger

    bars
      Either None (will be initialized with no bar) or a list/tuple of bar
      names (``['main', 'sub']``) which will be initialized with index -1 and
      no total, or a dictionary (possibly ordered) of bars, of the form
      ``{bar_1: {title: 'bar1', index: 2, total:23}, bar_2: {...}}``

    ignored_bars
      Either None (newly met bars will be added) or a list of blacklisted bar
      names, or ``'all_others'`` to signify that all bar names not already in
      ``self.bars`` will be ignored.


    leave_bars

    notebook
      True will make the bars look nice (HTML) in the jupyter notebook. It is
      advised to leave to 'default' as the default can be globally set from
      inside a notebook with ``import proglog; proglog.notebook_mode()``.

    print_messages
      If True, every ``logger(message='something')`` will print a message in
      the console / notebook
    """

    def __init__(self, init_state=None, bars=None, leave_bars=False, ignored_bars=None, logged_bars='all', notebook='default', print_messages=True, min_time_interval=0, ignore_bars_under=0):
        ProgressBarLogger.__init__(self, init_state=init_state, bars=bars, ignored_bars=ignored_bars, logged_bars=logged_bars, ignore_bars_under=ignore_bars_under, min_time_interval=min_time_interval)
        self.leave_bars = leave_bars
        self.tqdm_bars = OrderedDict([(bar, None) for bar in self.bars])
        if notebook == 'default':
            notebook = SETTINGS['notebook']
        self.notebook = notebook
        self.print_messages = print_messages
        self.tqdm = tqdm_notebook if self.notebook else tqdm

    def new_tqdm_bar(self, bar):
        """Create a new tqdm bar, possibly replacing an existing one."""
        if bar in self.tqdm_bars and self.tqdm_bars[bar] is not None:
            self.close_tqdm_bar(bar)
        infos = self.bars[bar]
        self.tqdm_bars[bar] = self.tqdm(total=infos['total'], desc=infos['title'], postfix=dict(now=troncate_string(str(infos['message']))), leave=self.leave_bars)

    def close_tqdm_bar(self, bar):
        """Close and erase the tqdm bar"""
        self.tqdm_bars[bar].close()
        if not self.notebook:
            self.tqdm_bars[bar] = None

    def bars_callback(self, bar, attr, value, old_value):
        if bar not in self.tqdm_bars or self.tqdm_bars[bar] is None:
            self.new_tqdm_bar(bar)
        if attr == 'index':
            if value >= old_value:
                total = self.bars[bar]['total']
                if total and value >= total:
                    self.close_tqdm_bar(bar)
                else:
                    self.tqdm_bars[bar].update(value - old_value)
            else:
                self.new_tqdm_bar(bar)
                self.tqdm_bars[bar].update(value + 1)
        elif attr == 'message':
            self.tqdm_bars[bar].set_postfix(now=troncate_string(str(value)))
            self.tqdm_bars[bar].update(0)

    def callback(self, **kw):
        if self.print_messages and 'message' in kw and kw['message']:
            if self.notebook:
                print(kw['message'])
            else:
                self.tqdm.write(kw['message'])