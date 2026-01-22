import pathlib
import param
from ...io.state import state
from ...theme import THEMES, DefaultTheme
from ...theme.fast import Design, Fast
from ..base import BasicTemplate
from ..react import ReactTemplate
@staticmethod
def _get_theme_from_query_args():
    theme_arg = state.session_args.get('theme', None)
    if not theme_arg:
        return
    theme_arg = theme_arg[0].decode('utf-8')
    return theme_arg.strip("'").strip('"')