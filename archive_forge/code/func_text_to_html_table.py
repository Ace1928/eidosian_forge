import re
import time
from typing import Optional
import IPython.display as disp
from ..trainer_callback import TrainerCallback
from ..trainer_utils import IntervalStrategy, has_length
def text_to_html_table(items):
    """Put the texts in `items` in an HTML table."""
    html_code = '<table border="1" class="dataframe">\n'
    html_code += '  <thead>\n <tr style="text-align: left;">\n'
    for i in items[0]:
        html_code += f'      <th>{i}</th>\n'
    html_code += '    </tr>\n  </thead>\n  <tbody>\n'
    for line in items[1:]:
        html_code += '    <tr>\n'
        for elt in line:
            elt = f'{elt:.6f}' if isinstance(elt, float) else str(elt)
            html_code += f'      <td>{elt}</td>\n'
        html_code += '    </tr>\n'
    html_code += '  </tbody>\n</table><p>'
    return html_code