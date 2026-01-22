import sys
def _display_progress_bar(self, size_read):
    if self._show_progress:
        self._percent += size_read / self._totalsize
        sys.stdout.write('\r[{0:<30}] {1:.0%}'.format('=' * int(round(self._percent * 29)) + '>', self._percent))
        sys.stdout.flush()