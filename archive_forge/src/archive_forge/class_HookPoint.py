from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
class HookPoint:
    """A single hook that clients can register to be called back when it fires.

    Attributes:
      name: The name of the hook.
      doc: The docs for using the hook.
      introduced: A version tuple specifying what version the hook was
                      introduced in. None indicates an unknown version.
      deprecated: A version tuple specifying what version the hook was
                      deprecated or superseded in. None indicates that the hook
                      is not superseded or deprecated. If the hook is
                      superseded then the doc should describe the recommended
                      replacement hook to register for.
    """

    def __init__(self, name, doc, introduced, deprecated=None, callbacks=None):
        """Create a HookPoint.

        :param name: The name of the hook, for clients to use when registering.
        :param doc: The docs for the hook.
        :param introduced: When the hook was introduced (e.g. (0, 15)).
        :param deprecated: When the hook was deprecated, None for
            not-deprecated.
        """
        self.name = name
        self.__doc__ = doc
        self.introduced = introduced
        self.deprecated = deprecated
        if callbacks is None:
            self._callbacks = []
        else:
            self._callbacks = callbacks

    def docs(self):
        """Generate the documentation for this HookPoint.

        :return: A string terminated in 
.
        """
        import textwrap
        strings = []
        strings.append(self.name)
        strings.append('~' * len(self.name))
        strings.append('')
        if self.introduced:
            introduced_string = _format_version_tuple(self.introduced)
        else:
            introduced_string = 'unknown'
        strings.append(gettext('Introduced in: %s') % introduced_string)
        if self.deprecated:
            deprecated_string = _format_version_tuple(self.deprecated)
            strings.append(gettext('Deprecated in: %s') % deprecated_string)
        strings.append('')
        strings.extend(textwrap.wrap(self.__doc__, break_long_words=False))
        strings.append('')
        return '\n'.join(strings)

    def __eq__(self, other):
        return isinstance(other, type(self)) and other.__dict__ == self.__dict__

    def hook_lazy(self, callback_module, callback_member, callback_label):
        """Lazily register a callback to be called when this HookPoint fires.

        :param callback_module: Module of the callable to use when this
            HookPoint fires.
        :param callback_member: Member name of the callback.
        :param callback_label: A label to show in the UI while this callback is
            processing.
        """
        obj_getter = registry._LazyObjectGetter(callback_module, callback_member)
        self._callbacks.append((obj_getter, callback_label))

    def hook(self, callback, callback_label):
        """Register a callback to be called when this HookPoint fires.

        :param callback: The callable to use when this HookPoint fires.
        :param callback_label: A label to show in the UI while this callback is
            processing.
        """
        obj_getter = registry._ObjectGetter(callback)
        self._callbacks.append((obj_getter, callback_label))

    def uninstall(self, label):
        """Uninstall the callback with the specified label.

        :param label: Label of the entry to uninstall
        """
        entries_to_remove = []
        for entry in self._callbacks:
            entry_callback, entry_label = entry
            if entry_label == label:
                entries_to_remove.append(entry)
        if entries_to_remove == []:
            raise KeyError('No entry with label %r' % label)
        for entry in entries_to_remove:
            self._callbacks.remove(entry)

    def __iter__(self):
        return (callback.get_obj() for callback, name in self._callbacks)

    def __len__(self):
        return len(self._callbacks)

    def __repr__(self):
        strings = []
        strings.append('<%s(' % type(self).__name__)
        strings.append(self.name)
        strings.append('), callbacks=[')
        callbacks = self._callbacks
        for callback, callback_name in callbacks:
            strings.append(repr(callback.get_obj()))
            strings.append('(')
            strings.append(callback_name)
            strings.append('),')
        if len(callbacks) == 1:
            strings[-1] = ')'
        strings.append(']>')
        return ''.join(strings)