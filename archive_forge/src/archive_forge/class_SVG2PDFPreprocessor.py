import base64
import os
import subprocess
import sys
from shutil import which
from tempfile import TemporaryDirectory
from traitlets import List, Unicode, Union, default
from nbconvert.utils.io import FormatSafeDict
from .convertfigures import ConvertFiguresPreprocessor
class SVG2PDFPreprocessor(ConvertFiguresPreprocessor):
    """
    Converts all of the outputs in a notebook from SVG to PDF.
    """

    @default('from_format')
    def _from_format_default(self):
        return 'image/svg+xml'

    @default('to_format')
    def _to_format_default(self):
        return 'application/pdf'
    inkscape_version = Unicode(help='The version of inkscape being used.\n\n        This affects how the conversion command is run.\n        ').tag(config=True)

    @default('inkscape_version')
    def _inkscape_version_default(self):
        p = subprocess.Popen([self.inkscape, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, _ = p.communicate()
        if p.returncode != 0:
            msg = 'Unable to find inkscape executable --version'
            raise RuntimeError(msg)
        return output.decode('utf-8').split(' ')[1]
    command = Union([Unicode(), List()], help='\n        The command to use for converting SVG to PDF\n\n        This traitlet is a template, which will be formatted with the keys\n        to_filename and from_filename.\n\n        The conversion call must read the SVG from {from_filename},\n        and write a PDF to {to_filename}.\n\n        It could be a List (recommended) or a String. If string, it will\n        be passed to a shell for execution.\n        ').tag(config=True)

    @default('command')
    def _command_default(self):
        major_version = self.inkscape_version.split('.')[0]
        command = [self.inkscape]
        if int(major_version) < 1:
            command.append('--without-gui')
            command.append('--export-pdf={to_filename}')
        else:
            command.append('--export-filename={to_filename}')
        command.append('{from_filename}')
        return command
    inkscape = Unicode(help='The path to Inkscape, if necessary').tag(config=True)

    @default('inkscape')
    def _inkscape_default(self):
        inkscape_path = which('inkscape')
        if inkscape_path is not None:
            return inkscape_path
        if sys.platform == 'darwin':
            if os.path.isfile(INKSCAPE_APP_v1):
                return INKSCAPE_APP_v1
            if os.path.isfile(INKSCAPE_APP):
                return INKSCAPE_APP
        if sys.platform == 'win32':
            wr_handle = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
            try:
                rkey = winreg.OpenKey(wr_handle, 'SOFTWARE\\Classes\\inkscape.svg\\DefaultIcon')
                inkscape = winreg.QueryValueEx(rkey, '')[0]
            except FileNotFoundError:
                msg = 'Inkscape executable not found'
                raise FileNotFoundError(msg) from None
            return inkscape
        return 'inkscape'

    def convert_figure(self, data_format, data):
        """
        Convert a single SVG figure to PDF.  Returns converted data.
        """
        with TemporaryDirectory() as tmpdir:
            input_filename = os.path.join(tmpdir, 'figure.svg')
            with open(input_filename, 'w', encoding='utf8') as f:
                f.write(data)
            output_filename = os.path.join(tmpdir, 'figure.pdf')
            template_vars = {'from_filename': input_filename, 'to_filename': output_filename}
            if isinstance(self.command, list):
                full_cmd = [s.format_map(FormatSafeDict(**template_vars)) for s in self.command]
            else:
                full_cmd = self.command.format(*template_vars)
            subprocess.call(full_cmd, shell=isinstance(full_cmd, str))
            if os.path.isfile(output_filename):
                with open(output_filename, 'rb') as f:
                    return base64.encodebytes(f.read()).decode('utf-8')
            else:
                msg = 'Inkscape svg to pdf conversion failed'
                raise TypeError(msg)