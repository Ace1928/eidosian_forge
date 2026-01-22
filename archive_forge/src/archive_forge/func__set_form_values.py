import os
import warnings
from typing import Optional, Sequence, Mapping, Callable
from ipywidgets import Dropdown, Text, Select, Button, HTML
from ipywidgets import Layout, GridBox, Box, HBox, VBox, ValueWidget
from .errors import ParentPathError, InvalidFileNameError
from .utils import get_subpaths, get_dir_contents, match_item, strip_parent_path
from .utils import is_valid_filename, get_drive_letters, normalize_path, has_parent_path
def _set_form_values(self, path: str, filename: str) -> None:
    """Set the form values."""
    if self._sandbox_path and (not has_parent_path(path, self._sandbox_path)):
        raise ParentPathError(path, self._sandbox_path)
    self._pathlist.unobserve(self._on_pathlist_select, names='value')
    self._dircontent.unobserve(self._on_dircontent_select, names='value')
    self._filename.unobserve(self._on_filename_change, names='value')
    try:
        _ = os.listdir(path)
        if self._show_only_dirs:
            filename = ''
        restricted_path = self._restrict_path(path)
        subpaths = get_subpaths(restricted_path)
        if os.path.splitdrive(subpaths[-1])[0]:
            drives = get_drive_letters()
            subpaths.extend(list(set(drives) - set(subpaths)))
        self._pathlist.options = subpaths
        self._pathlist.value = restricted_path
        self._filename.value = filename
        dircontent_real_names = get_dir_contents(path, show_hidden=self._show_hidden, show_only_dirs=self._show_only_dirs, dir_icon=None, filter_pattern=self._filter_pattern, top_path=self._sandbox_path)
        dircontent_display_names = get_dir_contents(path, show_hidden=self._show_hidden, show_only_dirs=self._show_only_dirs, dir_icon=self._dir_icon, dir_icon_append=self._dir_icon_append, filter_pattern=self._filter_pattern, top_path=self._sandbox_path)
        self._map_name_to_disp = {real_name: disp_name for real_name, disp_name in zip(dircontent_real_names, dircontent_display_names)}
        self._map_disp_to_name = {disp_name: real_name for real_name, disp_name in self._map_name_to_disp.items()}
        self._dircontent.options = dircontent_display_names
        if filename in dircontent_real_names and os.path.isfile(os.path.join(path, filename)):
            self._dircontent.value = self._map_name_to_disp[filename]
        else:
            self._dircontent.value = None
        if self._gb.layout.display is None:
            check1 = filename in dircontent_real_names
            check2 = os.path.isdir(os.path.join(path, filename))
            check3 = not is_valid_filename(filename)
            check4 = False
            check5 = False
            if self._selected_path is not None and self._selected_filename is not None:
                selected = os.path.join(self._selected_path, self._selected_filename)
                check4 = os.path.join(path, filename) == selected
            if self._filter_pattern:
                check5 = not match_item(filename, self._filter_pattern)
            if check1 and check2 or check3 or check4 or check5:
                self._select.disabled = True
            else:
                self._select.disabled = False
    except PermissionError:
        self._dircontent.value = None
        warnings.warn(f'Permission denied for {path}', RuntimeWarning)
    self._pathlist.observe(self._on_pathlist_select, names='value')
    self._dircontent.observe(self._on_dircontent_select, names='value')
    self._filename.observe(self._on_filename_change, names='value')