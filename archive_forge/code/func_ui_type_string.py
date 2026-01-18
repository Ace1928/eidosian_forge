import inspect
import re
import six
def ui_type_string(self, value=None, enum=False, reverse=False):
    """
        UI parameter type helper for string parameter type.
        @param value: Value to check against the type.
        @type value: anything
        @param enum: Has a meaning only if value is omitted. If set, returns
        a list of the possible values for the type, or [] if this is not
        possible. If not set, returns a text description of the type format.
        @type enum: bool
        @param reverse: If set, translates an internal value to its UI
        string representation.
        @type reverse: bool
        @return: c.f. parameter enum description.
        @rtype: str|list|None
        @raise ValueError: If the value does not check ok against the type.
        """
    if reverse:
        if value is not None:
            return value
        else:
            return 'n/a'
    type_enum = []
    syntax = 'STRING_OF_TEXT'
    if value is None:
        if enum:
            return type_enum
        else:
            return syntax
    elif not value:
        return None
    else:
        try:
            value = str(value)
        except ValueError:
            raise ValueError("Syntax error, '%s' is not a %s." % (value, syntax))
        else:
            return value