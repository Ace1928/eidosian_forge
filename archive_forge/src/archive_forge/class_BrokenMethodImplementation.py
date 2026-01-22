class BrokenMethodImplementation(_TargetInvalid):
    """
    BrokenMethodImplementation(method, message[, implementation, interface, target])

    The *target* (optional) has a *method* in *implementation* that violates
    its contract in a way described by *mess*.

    .. versionchanged:: 5.0.0
       Add the *interface* and *target* argument and attribute,
       and change the resulting string value of this object accordingly.

       The *method* can either be a simple string or a ``Method`` object.

    .. versionchanged:: 5.0.0
       If *implementation* is given, then the *message* will have the
       string "implementation" replaced with an short but informative
       representation of *implementation*.

    """
    _IX_IMPL = 2
    _IX_INTERFACE = _IX_IMPL + 1
    _IX_TARGET = _IX_INTERFACE + 1

    @property
    def method(self):
        return self.args[0]

    @property
    def mess(self):
        return self.args[1]

    @staticmethod
    def __implementation_str(impl):
        import inspect
        try:
            sig = inspect.signature
            formatsig = str
        except AttributeError:
            sig = inspect.getargspec
            f = inspect.formatargspec
            formatsig = lambda sig: f(*sig)
        try:
            sig = sig(impl)
        except (ValueError, TypeError):
            return repr(impl)
        try:
            name = impl.__qualname__
        except AttributeError:
            name = impl.__name__
        return name + formatsig(sig)

    @property
    def _str_details(self):
        impl = self._get_arg_or_default(self._IX_IMPL, self._NOT_GIVEN)
        message = self.mess
        if impl is not self._NOT_GIVEN and 'implementation' in message:
            message = message.replace('implementation', '%r')
            message = message % (self.__implementation_str(impl),)
        return 'The contract of {} is violated because {}'.format(repr(self.method) if isinstance(self.method, str) else self.method, message)