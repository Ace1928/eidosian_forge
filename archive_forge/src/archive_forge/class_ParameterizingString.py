import platform
import six
from blessed.colorspace import CGA_COLORS, X11_COLORNAMES_TO_RGB
class ParameterizingString(six.text_type):
    """
    A Unicode string which can be called as a parameterizing termcap.

    For example::

        >>> from blessed import Terminal
        >>> term = Terminal()
        >>> color = ParameterizingString(term.color, term.normal, 'color')
        >>> color(9)('color #9')
        u'\\x1b[91mcolor #9\\x1b(B\\x1b[m'
    """

    def __new__(cls, cap, normal=u'', name=u'<not specified>'):
        """
        Class constructor accepting 3 positional arguments.

        :arg str cap: parameterized string suitable for curses.tparm()
        :arg str normal: terminating sequence for this capability (optional).
        :arg str name: name of this terminal capability (optional).
        """
        new = six.text_type.__new__(cls, cap)
        new._normal = normal
        new._name = name
        return new

    def __call__(self, *args):
        """
        Returning :class:`FormattingString` instance for given parameters.

        Return evaluated terminal capability (self), receiving arguments
        ``*args``, followed by the terminating sequence (self.normal) into
        a :class:`FormattingString` capable of being called.

        :raises TypeError: Mismatch between capability and arguments
        :raises curses.error: :func:`curses.tparm` raised an exception
        :rtype: :class:`FormattingString` or :class:`NullCallableString`
        :returns: Callable string for given parameters
        """
        try:
            attr = curses.tparm(self.encode('latin1'), *args).decode('latin1')
            return FormattingString(attr, self._normal)
        except TypeError as err:
            if args and isinstance(args[0], six.string_types):
                raise TypeError('Unknown terminal capability, %r, or, TypeError for arguments %r: %s' % (self._name, args, err))
            raise
        except curses.error as err:
            if 'tparm() returned NULL' not in six.text_type(err):
                raise
            return NullCallableString()