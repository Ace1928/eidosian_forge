import re
import textwrap
import wrapt
from deprecated.classic import ClassicAdapter
from deprecated.classic import deprecated as _classic_deprecated
def versionchanged(reason='', version='', line_length=70):
    """
    This decorator can be used to insert a "versionchanged" directive
    in your function/class docstring in order to documents the
    version of the project which modifies this functionality in your library.

    :param str reason:
        Reason message which documents the modification in your library (can be omitted).

    :param str version:
        Version of your project which modifies this feature.
        If you follow the `Semantic Versioning <https://semver.org/>`_,
        the version number has the format "MAJOR.MINOR.PATCH".

    :type  line_length: int
    :param line_length:
        Max line length of the directive text. If non nul, a long text is wrapped in several lines.

    :return: the decorated function.
    """
    adapter = SphinxAdapter('versionchanged', reason=reason, version=version, line_length=line_length)
    return adapter