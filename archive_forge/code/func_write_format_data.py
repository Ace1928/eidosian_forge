from pygments.token import Token
import sys
from IPython.core.displayhook import DisplayHook
from prompt_toolkit.formatted_text import fragment_list_width, PygmentsTokens
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.enums import EditingMode
def write_format_data(self, format_dict, md_dict=None) -> None:
    assert self.shell is not None
    if self.shell.mime_renderers:
        for mime, handler in self.shell.mime_renderers.items():
            if mime in format_dict:
                handler(format_dict[mime], None)
                return
    super().write_format_data(format_dict, md_dict)