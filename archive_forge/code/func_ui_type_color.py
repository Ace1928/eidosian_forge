import inspect
import re
import six
def ui_type_color(self, value=None, enum=False, reverse=False):
    """
        UI parameter type helper for color parameter type.
        @param value: Value to check against the type.
        @type value: anything
        @param enum: Has a meaning only if value is omitted. If set, returns
        a list of the possible values for the type, or None if this is not
        possible. If not set, returns a text description of the type format.
        @type enum: bool
        @param reverse: If set, translates an internal value to its UI
        string representation.
        @type reverse: bool
        @return: c.f. parameter enum description.
        @rtype: str|list|None
        @raise ValueError: If the value does not check ok againts the type.
        """
    if reverse:
        if value is not None:
            return value
        else:
            return 'default'
    type_enum = self.shell.con.colors + ['default']
    syntax = '|'.join(type_enum)
    if value is None:
        if enum:
            return type_enum
        else:
            return syntax
    elif not value or value == 'default':
        return None
    elif value in type_enum:
        return value
    else:
        raise ValueError("Syntax error, '%s' is not %s" % (value, syntax))