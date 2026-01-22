import re
from sphinx.locale import _
from sphinx.ext.napoleon.docstring import NumpyDocstring
class InterfaceDocstring(NipypeDocstring):
    """
    Convert docstrings of Nipype Interfaces to reStructuredText.

    Parameters
    ----------
    docstring : :obj:`str` or :obj:`list` of :obj:`str`
        The docstring to parse, given either as a string or split into
        individual lines.
    config: :obj:`sphinx.ext.napoleon.Config` or :obj:`sphinx.config.Config`
        The configuration settings to use. If not given, defaults to the
        config object on `app`; or if `app` is not given defaults to the
        a new :class:`nipype.sphinxext.apidoc.Config` object.

    Other Parameters
    ----------------
    app : :class:`sphinx.application.Sphinx`, optional
        Application object representing the Sphinx process.
    what : :obj:`str`, optional
        A string specifying the type of the object to which the docstring
        belongs. Valid values: "module", "class", "exception", "function",
        "method", "attribute".
    name : :obj:`str`, optional
        The fully qualified name of the object.
    obj : module, class, exception, function, method, or attribute
        The object to which the docstring belongs.
    options : :class:`sphinx.ext.autodoc.Options`, optional
        The options given to the directive: an object with attributes
        inherited_members, undoc_members, show_inheritance and noindex that
        are True if the flag option of same name was given to the auto
        directive.

    """
    _name_rgx = re.compile('^\\s*(:(?P<role>\\w+):`(?P<name>[a-zA-Z0-9_.-]+)`| (?P<name2>[a-zA-Z0-9_.-]+))\\s*', re.X)

    def __init__(self, docstring, config=None, app=None, what='', name='', obj=None, options=None):
        super().__init__(docstring, config, app, what, name, obj, options)
        cmd = getattr(obj, '_cmd', '')
        if cmd and cmd.strip():
            self._parsed_lines = ['Wrapped executable: ``%s``.' % cmd.strip(), ''] + self._parsed_lines
        if obj is not None:
            self._parsed_lines += _parse_interface(obj)