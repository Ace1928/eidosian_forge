from __future__ import (absolute_import, division, print_function)
import os
from ansible import constants as C
from ansible.errors import AnsibleParserError
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.playbook.play import Play
from ansible.playbook.playbook_include import PlaybookInclude
from ansible.plugins.loader import add_all_plugin_dirs
from ansible.utils.display import Display
from ansible.utils.path import unfrackpath
def _load_playbook_data(self, file_name, variable_manager, vars=None):
    if os.path.isabs(file_name):
        self._basedir = os.path.dirname(file_name)
    else:
        self._basedir = os.path.normpath(os.path.join(self._basedir, os.path.dirname(file_name)))
    cur_basedir = self._loader.get_basedir()
    self._loader.set_basedir(self._basedir)
    add_all_plugin_dirs(self._basedir)
    self._file_name = file_name
    try:
        ds = self._loader.load_from_file(os.path.basename(file_name))
    except UnicodeDecodeError as e:
        raise AnsibleParserError('Could not read playbook (%s) due to encoding issues: %s' % (file_name, to_native(e)))
    if ds is None:
        self._loader.set_basedir(cur_basedir)
        raise AnsibleParserError('Empty playbook, nothing to do: %s' % unfrackpath(file_name), obj=ds)
    elif not isinstance(ds, list):
        self._loader.set_basedir(cur_basedir)
        raise AnsibleParserError('A playbook must be a list of plays, got a %s instead: %s' % (type(ds), unfrackpath(file_name)), obj=ds)
    elif not ds:
        self._loader.set_basedir(cur_basedir)
        raise AnsibleParserError('A playbook must contain at least one play: %s' % unfrackpath(file_name))
    for entry in ds:
        if not isinstance(entry, dict):
            self._loader.set_basedir(cur_basedir)
            raise AnsibleParserError("playbook entries must be either valid plays or 'import_playbook' statements", obj=entry)
        if any((action in entry for action in C._ACTION_IMPORT_PLAYBOOK)):
            pb = PlaybookInclude.load(entry, basedir=self._basedir, variable_manager=variable_manager, loader=self._loader)
            if pb is not None:
                self._entries.extend(pb._entries)
            else:
                which = entry
                for k in C._ACTION_IMPORT_PLAYBOOK:
                    if k in entry:
                        which = entry[k]
                        break
                display.display("skipping playbook '%s' due to conditional test failure" % which, color=C.COLOR_SKIP)
        else:
            entry_obj = Play.load(entry, variable_manager=variable_manager, loader=self._loader, vars=vars)
            self._entries.append(entry_obj)
    self._loader.set_basedir(cur_basedir)