from reportlab.rl_config import register_reset
class Sequencer:
    """Something to make it easy to number paragraphs, sections,
    images and anything else.  The features include registering
    new string formats for sequences, and 'chains' whereby
    some counters are reset when their parents.
    It keeps track of a number of
    'counters', which are created on request:
    Usage::
    
        >>> seq = layout.Sequencer()
        >>> seq.next('Bullets')
        1
        >>> seq.next('Bullets')
        2
        >>> seq.next('Bullets')
        3
        >>> seq.reset('Bullets')
        >>> seq.next('Bullets')
        1
        >>> seq.next('Figures')
        1
        >>>
    """

    def __init__(self):
        self._counters = {}
        self._formatters = {}
        self._reset()

    def _reset(self):
        self._counters.clear()
        self._formatters.clear()
        self._formatters.update({'1': _format_123, 'A': _format_ABC, 'a': _format_abc, 'I': _format_I, 'i': _format_i})
        d = dict(_counters=self._counters, _formatters=self._formatters)
        self.__dict__.clear()
        self.__dict__.update(d)
        self._defaultCounter = None

    def _getCounter(self, counter=None):
        """Creates one if not present"""
        try:
            return self._counters[counter]
        except KeyError:
            cnt = _Counter()
            self._counters[counter] = cnt
            return cnt

    def _this(self, counter=None):
        """Retrieves counter value but does not increment. For
        new counters, sets base value to 1."""
        if not counter:
            counter = self._defaultCounter
        return self._getCounter(counter)._this()

    def __next__(self):
        """Retrieves the numeric value for the given counter, then
        increments it by one.  New counters start at one."""
        return next(self._getCounter(self._defaultCounter))

    def next(self, counter=None):
        if not counter:
            return next(self)
        else:
            dc = self._defaultCounter
            try:
                self._defaultCounter = counter
                return next(self)
            finally:
                self._defaultCounter = dc

    def thisf(self, counter=None):
        if not counter:
            counter = self._defaultCounter
        return self._getCounter(counter).thisf()

    def nextf(self, counter=None):
        """Retrieves the numeric value for the given counter, then
        increments it by one.  New counters start at one."""
        if not counter:
            counter = self._defaultCounter
        return self._getCounter(counter).nextf()

    def setDefaultCounter(self, default=None):
        """Changes the key used for the default"""
        self._defaultCounter = default

    def registerFormat(self, format, func):
        """Registers a new formatting function.  The funtion
        must take a number as argument and return a string;
        fmt is a short menmonic string used to access it."""
        self._formatters[format] = func

    def setFormat(self, counter, format):
        """Specifies that the given counter should use
        the given format henceforth."""
        func = self._formatters[format]
        self._getCounter(counter).setFormatter(func)

    def reset(self, counter=None, base=0):
        if not counter:
            counter = self._defaultCounter
        self._getCounter(counter)._value = base

    def chain(self, parent, child):
        p = self._getCounter(parent)
        c = self._getCounter(child)
        p.chain(c)

    def __getitem__(self, key):
        """Allows compact notation to support the format function.
        s['key'] gets current value, s['key+'] increments."""
        if key[-1:] == '+':
            counter = key[:-1]
            return self.nextf(counter)
        else:
            return self.thisf(key)

    def format(self, template):
        """The crowning jewels - formats multi-level lists."""
        return template % self

    def dump(self):
        """Write current state to stdout for diagnostics"""
        counters = list(self._counters.items())
        counters.sort()
        print('Sequencer dump:')
        for key, counter in counters:
            print('    %s: value = %d, base = %d, format example = %s' % (key, counter._this(), counter._base, counter.thisf()))