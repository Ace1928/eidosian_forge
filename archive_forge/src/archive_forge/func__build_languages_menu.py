import enchant
import gettext
import logging
import re
import sys
from collections import UserList
from ._pylocales import code_to_name as _code_to_name
from ._pylocales import LanguageNotFound, CountryNotFound
from gi.repository import Gio, GLib, GObject
def _build_languages_menu(self):
    if _IS_GTK3:

        def _set_language(item, code):
            self.language = code
        menu = Gtk.Menu.new()
        group = []
        connect = []
    else:
        menu = Gio.Menu.new()
    for code, name in self.languages:
        if _IS_GTK3:
            item = Gtk.RadioMenuItem.new_with_label(group, name)
            group.append(item)
            if code == self.language:
                item.set_active(True)
            connect.append((item, code))
            menu.append(item)
        else:
            item = Gio.MenuItem.new(name, None)
            item.set_action_and_target_value('spelling.language', GLib.Variant.new_string(code))
            menu.append_item(item)
    if _IS_GTK3:
        for item, code in connect:
            item.connect('activate', _set_language, code)
        return menu
    else:
        return Gio.MenuItem.new_submenu(_('Languages'), menu)