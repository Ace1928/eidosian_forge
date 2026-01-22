Convenience function for configuring _() to use lazy gettext

    Call this at the start of execution to enable the gettextutils._
    function to use lazy gettext functionality. This is useful if
    your project is importing _ directly instead of using the
    gettextutils.install() way of importing the _ function.

    :param enable: Flag indicating whether lazy translation should be
                   turned on or off.  Defaults to True.
    :type enable: bool

    