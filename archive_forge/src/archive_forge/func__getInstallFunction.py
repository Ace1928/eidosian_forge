from twisted.python.runtime import platform
def _getInstallFunction(platform):
    """
    Return a function to install the reactor most suited for the given platform.

    @param platform: The platform for which to select a reactor.
    @type platform: L{twisted.python.runtime.Platform}

    @return: A zero-argument callable which will install the selected
        reactor.
    """
    try:
        if platform.isLinux():
            try:
                from twisted.internet.epollreactor import install
            except ImportError:
                from twisted.internet.pollreactor import install
        elif platform.getType() == 'posix' and (not platform.isMacOSX()):
            from twisted.internet.pollreactor import install
        else:
            from twisted.internet.selectreactor import install
    except ImportError:
        from twisted.internet.selectreactor import install
    return install