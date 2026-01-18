from __future__ import annotations
import re
from fractions import Fraction
def to_html_string(self) -> str:
    """Generates a HTML formatted string. This uses the output from to_latex_string to generate a HTML output.

        Returns:
            HTML formatted string.
        """
    str_ = re.sub('\\$_\\{([^}]+)\\}\\$', '<sub>\\1</sub>', self.to_latex_string())
    str_ = re.sub('\\$\\^\\{([^}]+)\\}\\$', '<sup>\\1</sup>', str_)
    return re.sub('\\$\\\\overline\\{([^}]+)\\}\\$', '<span style="text-decoration:overline">\\1</span>', str_)