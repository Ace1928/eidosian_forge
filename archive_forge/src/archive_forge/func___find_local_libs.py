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
def __find_local_libs(self, dependency_files: List[zipfile.Path]):
    local_libs = set()
    plugins = set()
    lib_pattern = re.compile(f'lib(?P<lib_name>.*)_{self.arch}')
    for dependency_file in dependency_files:
        xml_content = dependency_file.read_text()
        root = ET.fromstring(xml_content)
        for local_lib in root.iter('lib'):
            if 'file' not in local_lib.attrib:
                if 'name' not in local_lib.attrib:
                    logging.warning(f'[DEPLOY] Invalid android dependency file {str(dependency_file)}')
                continue
            file = local_lib.attrib['file']
            if file.endswith('.so'):
                file_name = Path(file).stem
                if file_name.startswith('libplugins_platforms_qtforandroid'):
                    continue
                match = lib_pattern.search(file_name)
                if match:
                    lib_name = match.group('lib_name')
                    local_libs.add(lib_name)
                    if lib_name.startswith('plugins'):
                        plugin_name = lib_name.split('plugins_', 1)[1]
                        plugins.add(plugin_name)
    return (list(local_libs), list(plugins))