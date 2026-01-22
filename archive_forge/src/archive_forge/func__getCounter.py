from reportlab.rl_config import register_reset
def _getCounter(self, counter=None):
    """Creates one if not present"""
    try:
        return self._counters[counter]
    except KeyError:
        cnt = _Counter()
        self._counters[counter] = cnt
        return cnt