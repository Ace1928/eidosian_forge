from reportlab.rl_config import register_reset
def _this(self, counter=None):
    """Retrieves counter value but does not increment. For
        new counters, sets base value to 1."""
    if not counter:
        counter = self._defaultCounter
    return self._getCounter(counter)._this()