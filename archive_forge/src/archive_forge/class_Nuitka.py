import logging
import os
import sys
from pathlib import Path
from typing import List
from . import MAJOR_VERSION, run_command
class Nuitka:
    """
    Wrapper class around the nuitka executable, enabling its usage through python code
    """

    def __init__(self, nuitka):
        self.nuitka = nuitka

    @staticmethod
    def icon_option():
        if sys.platform == 'linux':
            return '--linux-icon'
        elif sys.platform == 'win32':
            return '--windows-icon-from-ico'
        else:
            return '--macos-app-icon'

    def create_executable(self, source_file: Path, extra_args: str, qml_files: List[Path], excluded_qml_plugins: List[str], icon: str, dry_run: bool):
        extra_args = extra_args.split()
        qml_args = []
        if qml_files:
            qml_args.append('--include-qt-plugins=all')
            qml_args.extend([f'--include-data-files={qml_file.resolve()}=./{qml_file.resolve().relative_to(source_file.parent)}' for qml_file in qml_files])
            if excluded_qml_plugins:
                prefix = 'lib' if sys.platform != 'win32' else ''
                for plugin in excluded_qml_plugins:
                    dll_name = plugin.replace('Qt', f'Qt{MAJOR_VERSION}')
                    qml_args.append(f'--noinclude-dlls={prefix}{dll_name}*')
        output_dir = source_file.parent / 'deployment'
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info('[DEPLOY] Running Nuitka')
        command = self.nuitka + [os.fspath(source_file), '--follow-imports', '--onefile', '--enable-plugin=pyside6', f'--output-dir={output_dir}']
        command.extend(extra_args + qml_args)
        command.append(f'{self.__class__.icon_option()}={icon}')
        command_str, _ = run_command(command=command, dry_run=dry_run)
        return command_str