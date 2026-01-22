from __future__ import annotations
import itertools
import typing
import warnings
import weakref
class Signals:
    _signal_attr = '_urwid_signals'

    def __init__(self) -> None:
        self._supported = {}

    def register(self, sig_cls, signals: Container[Hashable]) -> None:
        """
        :param sig_cls: the class of an object that will be sending signals
        :type sig_cls: class
        :param signals: a list of signals that may be sent, typically each
                        signal is represented by a string
        :type signals: signal names

        This function must be called for a class before connecting any
        signal callbacks or emitting any signals from that class' objects
        """
        self._supported[sig_cls] = signals

    def connect(self, obj, name: Hashable, callback: Callable[..., typing.Any], user_arg: typing.Any=None, *, weak_args: Iterable[typing.Any]=(), user_args: Iterable[typing.Any]=()) -> Key:
        """
        :param obj: the object sending a signal
        :type obj: object
        :param name: the signal to listen for, typically a string
        :type name: signal name
        :param callback: the function to call when that signal is sent
        :type callback: function
        :param user_arg: deprecated additional argument to callback (appended
                         after the arguments passed when the signal is
                         emitted). If None no arguments will be added.
                         Don't use this argument, use user_args instead.
        :param weak_args: additional arguments passed to the callback
                          (before any arguments passed when the signal
                          is emitted and before any user_args).

                          These arguments are stored as weak references
                          (but converted back into their original value
                          before passing them to callback) to prevent
                          any objects referenced (indirectly) from
                          weak_args from being kept alive just because
                          they are referenced by this signal handler.

                          Use this argument only as a keyword argument,
                          since user_arg might be removed in the future.
        :type weak_args: iterable
        :param user_args: additional arguments to pass to the callback,
                          (before any arguments passed when the signal
                          is emitted but after any weak_args).

                          Use this argument only as a keyword argument,
                          since user_arg might be removed in the future.
        :type user_args: iterable

        When a matching signal is sent, callback will be called. The
        arguments it receives will be the user_args passed at connect
        time (as individual arguments) followed by all the positional
        parameters sent with the signal.

        As an example of using weak_args, consider the following snippet:

        >>> import urwid
        >>> debug = urwid.Text('')
        >>> def handler(widget, newtext):
        ...    debug.set_text("Edit widget changed to %s" % newtext)
        >>> edit = urwid.Edit('')
        >>> key = urwid.connect_signal(edit, 'change', handler)

        If you now build some interface using "edit" and "debug", the
        "debug" widget will show whatever you type in the "edit" widget.
        However, if you remove all references to the "debug" widget, it
        will still be kept alive by the signal handler. This because the
        signal handler is a closure that (implicitly) references the
        "edit" widget. If you want to allow the "debug" widget to be
        garbage collected, you can create a "fake" or "weak" closure
        (it's not really a closure, since it doesn't reference any
        outside variables, so it's just a dynamic function):

        >>> debug = urwid.Text('')
        >>> def handler(weak_debug, widget, newtext):
        ...    weak_debug.set_text("Edit widget changed to %s" % newtext)
        >>> edit = urwid.Edit('')
        >>> key = urwid.connect_signal(edit, 'change', handler, weak_args=[debug])

        Here the weak_debug parameter in print_debug is the value passed
        in the weak_args list to connect_signal. Note that the
        weak_debug value passed is not a weak reference anymore, the
        signals code transparently dereferences the weakref parameter
        before passing it to print_debug.

        Returns a key associated by this signal handler, which can be
        used to disconnect the signal later on using
        urwid.disconnect_signal_by_key. Alternatively, the signal
        handler can also be disconnected by calling
        urwid.disconnect_signal, which doesn't need this key.
        """
        if user_arg is not None:
            warnings.warn("Don't use user_arg argument, use user_args instead.", DeprecationWarning, stacklevel=2)
        sig_cls = obj.__class__
        if name not in self._supported.get(sig_cls, ()):
            raise NameError(f'No such signal {name!r} for object {obj!r}')
        key = Key()
        handlers = setdefaultattr(obj, self._signal_attr, {}).setdefault(name, [])
        obj_weak = weakref.ref(obj)

        def weakref_callback(weakref):
            o = obj_weak()
            if o:
                self.disconnect_by_key(o, name, key)
        user_args = self._prepare_user_args(weak_args, user_args, weakref_callback)
        handlers.append((key, callback, user_arg, user_args))
        return key

    def _prepare_user_args(self, weak_args: Iterable[typing.Any]=(), user_args: Iterable[typing.Any]=(), callback: Callable[..., typing.Any] | None=None) -> tuple[Collection[weakref.ReferenceType], Collection[typing.Any]]:
        w_args = tuple((weakref.ref(w_arg, callback) for w_arg in weak_args))
        args = tuple(user_args) or ()
        return (w_args, args)

    def disconnect(self, obj, name: Hashable, callback: Callable[..., typing.Any], user_arg: typing.Any=None, *, weak_args: Iterable[typing.Any]=(), user_args: Iterable[typing.Any]=()) -> None:
        """
        :param obj: the object to disconnect the signal from
        :type obj: object
        :param name: the signal to disconnect, typically a string
        :type name: signal name
        :param callback: the callback function passed to connect_signal
        :type callback: function
        :param user_arg: the user_arg parameter passed to connect_signal
        :param weak_args: the weak_args parameter passed to connect_signal
        :param user_args: the weak_args parameter passed to connect_signal

        This function will remove a callback from the list connected
        to a signal with connect_signal(). The arguments passed should
        be exactly the same as those passed to connect_signal().

        If the callback is not connected or already disconnected, this
        function will simply do nothing.
        """
        signals = setdefaultattr(obj, self._signal_attr, {})
        if name not in signals:
            return None
        handlers = signals[name]
        user_args = self._prepare_user_args(weak_args, user_args)
        for h in handlers:
            if h[1:] == (callback, user_arg, user_args):
                return self.disconnect_by_key(obj, name, h[0])
        return None

    def disconnect_by_key(self, obj, name: Hashable, key: Key) -> None:
        """
        :param obj: the object to disconnect the signal from
        :type obj: object
        :param name: the signal to disconnect, typically a string
        :type name: signal name
        :param key: the key for this signal handler, as returned by
                    connect_signal().
        :type key: Key

        This function will remove a callback from the list connected
        to a signal with connect_signal(). The key passed should be the
        value returned by connect_signal().

        If the callback is not connected or already disconnected, this
        function will simply do nothing.
        """
        handlers = setdefaultattr(obj, self._signal_attr, {}).get(name, [])
        handlers[:] = [h for h in handlers if h[0] is not key]

    def emit(self, obj, name: Hashable, *args) -> bool:
        """
        :param obj: the object sending a signal
        :type obj: object
        :param name: the signal to send, typically a string
        :type name: signal name
        :param args: zero or more positional arguments to pass to the signal callback functions

        This function calls each of the callbacks connected to this signal
        with the args arguments as positional parameters.

        This function returns True if any of the callbacks returned True.
        """
        result = False
        handlers = getattr(obj, self._signal_attr, {}).get(name, [])
        for _key, callback, user_arg, (weak_args, user_args) in handlers:
            result |= self._call_callback(callback, user_arg, weak_args, user_args, args)
        return result

    def _call_callback(self, callback, user_arg: typing.Any, weak_args: Collection[weakref.ReferenceType], user_args: Collection[typing.Any], emit_args: Iterable[typing.Any]) -> bool:
        args_to_pass = []
        for w_arg in weak_args:
            real_arg = w_arg()
            if real_arg is not None:
                args_to_pass.append(real_arg)
            else:
                return False
        args = itertools.chain(args_to_pass, user_args, emit_args, (user_arg,) if user_arg is not None else ())
        return bool(callback(*args))