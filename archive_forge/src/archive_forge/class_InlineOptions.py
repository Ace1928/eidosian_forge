from abc import ABCMeta, abstractmethod
class InlineOptions(AbstractOptionValue):
    """
    Options for controlling inlining
    """

    def __init__(self, value):
        ok = False
        if isinstance(value, str):
            if value in ('always', 'never'):
                ok = True
        else:
            ok = hasattr(value, '__call__')
        if ok:
            self._inline = value
        else:
            msg = "kwarg 'inline' must be one of the strings 'always' or 'never', or it can be a callable that returns True/False. Found value %s" % value
            raise ValueError(msg)

    @property
    def is_never_inline(self):
        """
        True if never inline
        """
        return self._inline == 'never'

    @property
    def is_always_inline(self):
        """
        True if always inline
        """
        return self._inline == 'always'

    @property
    def has_cost_model(self):
        """
        True if a cost model is provided
        """
        return not (self.is_always_inline or self.is_never_inline)

    @property
    def value(self):
        """
        The raw value
        """
        return self._inline

    def __eq__(self, other):
        if type(other) is type(self):
            return self.value == other.value
        return NotImplemented

    def encode(self) -> str:
        return repr(self._inline)