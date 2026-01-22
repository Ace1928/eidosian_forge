from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console.style import ansi
from googlecloudsdk.core.console.style import text
def GetStyleMappings(console_attributes=None):
    """Gets the style mappings based on the console and user properties."""
    console_attributes = console_attributes or console_attr.GetConsoleAttr()
    is_screen_reader = properties.VALUES.accessibility.screen_reader.GetBool()
    if properties.VALUES.core.color_theme.Get() == 'testing':
        return STYLE_MAPPINGS_TESTING
    elif not is_screen_reader and console_attributes.SupportsAnsi() and (properties.VALUES.core.color_theme.Get() != 'off'):
        if console_attributes._term == 'xterm-256color':
            return STYLE_MAPPINGS_ANSI_256
        else:
            return STYLE_MAPPINGS_ANSI
    else:
        return STYLE_MAPPINGS_BASIC