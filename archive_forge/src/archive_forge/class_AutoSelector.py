from __future__ import unicode_literals, absolute_import
import sys
import abc
import errno
import select
import six
class AutoSelector(Selector):

    def __init__(self):
        self._fds = []
        self._select_selector = SelectSelector()
        self._selectors = [self._select_selector]
        if hasattr(select, 'poll'):
            self._poll_selector = PollSelector()
            self._selectors.append(self._poll_selector)
        else:
            self._poll_selector = None
        if sys.version_info >= (3, 5):
            self._py3_selector = Python3Selector()
            self._selectors.append(self._py3_selector)
        else:
            self._py3_selector = None

    def register(self, fd):
        assert isinstance(fd, int)
        self._fds.append(fd)
        for sel in self._selectors:
            sel.register(fd)

    def unregister(self, fd):
        assert isinstance(fd, int)
        self._fds.remove(fd)
        for sel in self._selectors:
            sel.unregister(fd)

    def select(self, timeout):
        if self._py3_selector:
            try:
                return self._py3_selector.select(timeout)
            except PermissionError:
                pass
        try:
            return self._select_selector.select(timeout)
        except ValueError:
            if self._poll_selector is not None:
                return self._poll_selector.select(timeout)
            else:
                raise

    def close(self):
        for sel in self._selectors:
            sel.close()