from __future__ import annotations
import re
from fractions import Fraction
class Stringify:
    """Mix-in class for string formatting, e.g. superscripting numbers and symbols or superscripting."""
    STRING_MODE = 'SUBSCRIPT'

    def to_pretty_string(self) -> str:
        """A pretty string representation. By default, the __str__ output is used, but this method can be
        overridden if a different representation from default is desired.
        """
        return str(self)

    def to_latex_string(self) -> str:
        """Generates a LaTeX formatted string. The mode is set by the class variable STRING_MODE, which defaults to
        "SUBSCRIPT". E.g., Fe2O3 is transformed to Fe$_{2}$O$_{3}$. Setting STRING_MODE to "SUPERSCRIPT" creates
        superscript, e.g., Fe2+ becomes Fe^{2+}. The initial string is obtained from the class's __str__ method.

        Returns:
            String for display as in LaTeX with proper superscripts and subscripts.
        """
        str_ = self.to_pretty_string()
        str_ = re.sub('_(\\d+)', '$_{\\1}$', str_)
        str_ = re.sub('\\^([\\d\\+\\-]+)', '$^{\\1}$', str_)
        if self.STRING_MODE == 'SUBSCRIPT':
            return re.sub('([A-Za-z\\(\\)])([\\d\\+\\-\\.]+)', '\\1$_{\\2}$', str_)
        if self.STRING_MODE == 'SUPERSCRIPT':
            return re.sub('([A-Za-z\\(\\)])([\\d\\+\\-\\.]+)', '\\1$^{\\2}$', str_)
        return str_

    def to_html_string(self) -> str:
        """Generates a HTML formatted string. This uses the output from to_latex_string to generate a HTML output.

        Returns:
            HTML formatted string.
        """
        str_ = re.sub('\\$_\\{([^}]+)\\}\\$', '<sub>\\1</sub>', self.to_latex_string())
        str_ = re.sub('\\$\\^\\{([^}]+)\\}\\$', '<sup>\\1</sup>', str_)
        return re.sub('\\$\\\\overline\\{([^}]+)\\}\\$', '<span style="text-decoration:overline">\\1</span>', str_)

    def to_unicode_string(self):
        """Unicode string with proper sub and superscripts. Note that this works only
        with systems where the sub and superscripts are pure integers.
        """
        str_ = self.to_latex_string()
        for m in re.finditer('\\$_\\{(\\d+)\\}\\$', str_):
            s1 = m.group()
            s2 = [SUBSCRIPT_UNICODE[s] for s in m.group(1)]
            str_ = str_.replace(s1, ''.join(s2))
        for m in re.finditer('\\$\\^\\{([\\d\\+\\-]+)\\}\\$', str_):
            s1 = m.group()
            s2 = [SUPERSCRIPT_UNICODE[s] for s in m.group(1)]
            str_ = str_.replace(s1, ''.join(s2))
        return str_