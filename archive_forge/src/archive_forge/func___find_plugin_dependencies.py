import sys
import logging
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
import PySide6
from pathlib import Path
from typing import List
from pkginfo import Wheel
from .. import MAJOR_VERSION, BaseConfig, Config, run_command
from . import (create_recipe, find_lib_dependencies, find_qtlibs_in_wheel,
def __find_plugin_dependencies(self, dependency_files: List[zipfile.Path], dependent_plugins: List[str]):
    lib_pattern = re.compile(f'libplugins_(?P<plugin_name>.*)_{self.arch}.so')
    for dependency_file in dependency_files:
        xml_content = dependency_file.read_text()
        root = ET.fromstring(xml_content)
        for bundled_element in root.iter('bundled'):
            if 'file' not in bundled_element.attrib:
                logging.warning(f'[DEPLOY] Invalid Android dependency file {str(dependency_file)}')
                continue
            plugin_module_folder = bundled_element.attrib['file']
            if plugin_module_folder.startswith('./plugins'):
                plugin_module_folder = plugin_module_folder.partition('./plugins/')[2]
            else:
                continue
            absolute_plugin_module_folder = self.qt_libs_path.parent / 'plugins' / plugin_module_folder
            if not absolute_plugin_module_folder.is_dir():
                logging.warning(f"[DEPLOY] Qt plugin folder '{plugin_module_folder}' does not exist or is not a directory for this Android platform")
                continue
            for plugin in absolute_plugin_module_folder.iterdir():
                plugin_name = plugin.name
                if plugin_name.endswith('.so') and plugin_name.startswith('libplugins'):
                    match = lib_pattern.search(plugin_name)
                    if match:
                        plugin_infix_name = match.group('plugin_name')
                        if plugin_infix_name not in dependent_plugins:
                            dependent_plugins.append(plugin_infix_name)