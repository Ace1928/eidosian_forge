import inspect
import re
import six
def ui_getgroup_global(self, parameter):
    """
        This is the backend method for getting configuration parameters out of
        the global configuration group. It gets the values from the Prefs()
        backend. Eventual casting to str for UI display is handled by the ui
        get command, for symmetry with the pendant ui_setgroup method.
        Existence of the parameter in the group should have already been
        checked by the ui get command, so we go blindly about this. This might
        allow internal client code to get a None value if the parameter does
        not exist, as supported by Prefs().

        @param parameter: The parameter to get the value of.
        @type parameter: str
        @return: The parameter's value
        @rtype: arbitrary
        """
    return self.shell.prefs[parameter]