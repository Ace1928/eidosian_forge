from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
import re
from typing import (
from uuid import uuid4
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import ABCSeries
from pandas import (
from pandas.api.types import is_list_like
import pandas.core.common as com
from markupsafe import escape as escape_html  # markupsafe is jinja2 dependency
def _parse_latex_css_conversion(styles: CSSList) -> CSSList:
    """
    Convert CSS (attribute,value) pairs to equivalent LaTeX (command,options) pairs.

    Ignore conversion if tagged with `--latex` option, skipped if no conversion found.
    """

    def font_weight(value, arg):
        if value in ('bold', 'bolder'):
            return ('bfseries', f'{arg}')
        return None

    def font_style(value, arg):
        if value == 'italic':
            return ('itshape', f'{arg}')
        if value == 'oblique':
            return ('slshape', f'{arg}')
        return None

    def color(value, user_arg, command, comm_arg):
        """
        CSS colors have 5 formats to process:

         - 6 digit hex code: "#ff23ee"     --> [HTML]{FF23EE}
         - 3 digit hex code: "#f0e"        --> [HTML]{FF00EE}
         - rgba: rgba(128, 255, 0, 0.5)    --> [rgb]{0.502, 1.000, 0.000}
         - rgb: rgb(128, 255, 0,)          --> [rbg]{0.502, 1.000, 0.000}
         - string: red                     --> {red}

        Additionally rgb or rgba can be expressed in % which is also parsed.
        """
        arg = user_arg if user_arg != '' else comm_arg
        if value[0] == '#' and len(value) == 7:
            return (command, f'[HTML]{{{value[1:].upper()}}}{arg}')
        if value[0] == '#' and len(value) == 4:
            val = f'{value[1].upper() * 2}{value[2].upper() * 2}{value[3].upper() * 2}'
            return (command, f'[HTML]{{{val}}}{arg}')
        elif value[:3] == 'rgb':
            r = re.findall('(?<=\\()[0-9\\s%]+(?=,)', value)[0].strip()
            r = float(r[:-1]) / 100 if '%' in r else int(r) / 255
            g = re.findall('(?<=,)[0-9\\s%]+(?=,)', value)[0].strip()
            g = float(g[:-1]) / 100 if '%' in g else int(g) / 255
            if value[3] == 'a':
                b = re.findall('(?<=,)[0-9\\s%]+(?=,)', value)[1].strip()
            else:
                b = re.findall('(?<=,)[0-9\\s%]+(?=\\))', value)[0].strip()
            b = float(b[:-1]) / 100 if '%' in b else int(b) / 255
            return (command, f'[rgb]{{{r:.3f}, {g:.3f}, {b:.3f}}}{arg}')
        else:
            return (command, f'{{{value}}}{arg}')
    CONVERTED_ATTRIBUTES: dict[str, Callable] = {'font-weight': font_weight, 'background-color': partial(color, command='cellcolor', comm_arg='--lwrap'), 'color': partial(color, command='color', comm_arg=''), 'font-style': font_style}
    latex_styles: CSSList = []
    for attribute, value in styles:
        if isinstance(value, str) and '--latex' in value:
            latex_styles.append((attribute, value.replace('--latex', '')))
        if attribute in CONVERTED_ATTRIBUTES:
            arg = ''
            for x in ['--wrap', '--nowrap', '--lwrap', '--dwrap', '--rwrap']:
                if x in str(value):
                    arg, value = (x, _parse_latex_options_strip(value, x))
                    break
            latex_style = CONVERTED_ATTRIBUTES[attribute](value, arg)
            if latex_style is not None:
                latex_styles.extend([latex_style])
    return latex_styles