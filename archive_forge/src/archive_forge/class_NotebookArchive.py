import os
import sys
import time
import traceback
import param
from IPython.display import Javascript, display
from nbconvert import HTMLExporter, NotebookExporter
from nbconvert.preprocessors.clearoutput import ClearOutputPreprocessor
from nbformat import reader
from ..core.io import FileArchive, Pickler
from ..plotting.renderer import HTML_TAGS, MIME_TYPES
from .preprocessors import Substitute
class NotebookArchive(FileArchive):
    """
    FileArchive that can automatically capture notebook data via the
    display hooks and automatically adds a notebook HTML snapshot to
    the archive upon export.
    """
    exporters = param.List(default=[Pickler])
    skip_notebook_export = param.Boolean(default=False, doc='\n        Whether to skip JavaScript capture of notebook data which may\n        be unreliable. Also disabled automatic capture of notebook\n        name.')
    snapshot_name = param.String('index', doc='\n        The basename of the exported notebook snapshot (html). It may\n        optionally use the {timestamp} formatter.')
    filename_formatter = param.String(default='{dimensions},{obj}', doc='\n        Similar to FileArchive.filename_formatter except with support\n        for the notebook name field as {notebook}.')
    export_name = param.String(default='{notebook}', doc='\n        Similar to FileArchive.filename_formatter except with support\n        for the notebook name field as {notebook}.')
    auto = param.Boolean(False)
    traceback = None
    ffields = FileArchive.ffields.union({'notebook'})
    efields = FileArchive.efields.union({'notebook'})

    def __init__(self, **params):
        super().__init__(**params)
        self.nbversion = None
        self.notebook_name = None
        self.export_success = None
        self._auto = False
        self._replacements = {}
        self._notebook_data = None
        self._timestamp = None
        self._tags = {MIME_TYPES[k]: v for k, v in HTML_TAGS.items() if k in MIME_TYPES}
        keywords = [f'{k}={v.__class__.__name__}' for k, v in self.param.objects().items()]
        self.auto.__func__.__doc__ = f'auto(enabled=Boolean, {', '.join(keywords)})'

    def get_namespace(self):
        """
        Find the name the user is using to access holoviews.
        """
        if 'holoviews' not in sys.modules:
            raise ImportError('HoloViews does not seem to be imported')
        matches = [k for k, v in get_ipython().user_ns.items() if not k.startswith('_') and v is sys.modules['holoviews']]
        if len(matches) == 0:
            raise Exception('Could not find holoviews module in namespace')
        return f'{matches[0]}.archive'

    def last_export_status(self):
        """Helper to show the status of the last call to the export method."""
        if self.export_success is True:
            print('The last call to holoviews.archive.export was successful.')
            return
        elif self.export_success is None:
            print('Status of the last call to holoviews.archive.export is unknown.\n(Re-execute this method once kernel status is idle.)')
            return
        print('The last call to holoviews.archive.export was unsuccessful.')
        if self.traceback is None:
            print('\n<No traceback captured>')
        else:
            print('\n' + self.traceback)

    def auto(self, enabled=True, clear=False, **kwargs):
        """
        Method to enable or disable automatic capture, allowing you to
        simultaneously set the instance parameters.
        """
        self.namespace = self.get_namespace()
        self.notebook_name = '{notebook}'
        self._timestamp = tuple(time.localtime())
        kernel = 'var kernel = IPython.notebook.kernel; '
        nbname = 'var nbname = IPython.notebook.get_notebook_name(); '
        nbcmd = 'var name_cmd = \'%s.notebook_name = \\"\' + nbname + \'\\"\'; ' % self.namespace
        cmd = kernel + nbname + nbcmd + 'kernel.execute(name_cmd); '
        display(Javascript(cmd))
        time.sleep(0.5)
        self._auto = enabled
        self.param.update(**kwargs)
        tstamp = time.strftime(' [%Y-%m-%d %H:%M:%S]', self._timestamp)
        if clear:
            FileArchive.clear(self)
        print('Automatic capture is now {}.{}'.format('enabled' if enabled else 'disabled', tstamp if enabled else ''))

    def export(self, timestamp=None):
        """
        Get the current notebook data and export.
        """
        if self._timestamp is None:
            raise Exception('No timestamp set. Has the archive been initialized?')
        if self.skip_notebook_export:
            super().export(timestamp=self._timestamp, info={'notebook': self.notebook_name})
            return
        self.export_success = None
        name = self.get_namespace()
        capture_cmd = 'var capture = \'%s._notebook_data=r\\"\\"\\"\'' % name + '+json_string+\'\\"\\"\\"\'; '
        cmd = 'var kernel = IPython.notebook.kernel; ' + 'var json_data = IPython.notebook.toJSON(); ' + 'var json_string = JSON.stringify(json_data); ' + capture_cmd + f"var pycmd = capture + ';{name}._export_with_html()'; " + 'kernel.execute(pycmd)'
        tstamp = time.strftime(self.timestamp_format, self._timestamp)
        export_name = self._format(self.export_name, {'timestamp': tstamp, 'notebook': self.notebook_name})
        print('Export name: {!r}\nDirectory    {!r}'.format(export_name, os.path.join(os.path.abspath(self.root))) + '\n\nIf no output appears, please check holoviews.archive.last_export_status()')
        display(Javascript(cmd))

    def add(self, obj=None, filename=None, data=None, info=None, html=None):
        """Similar to FileArchive.add but accepts html strings for substitution"""
        if info is None:
            info = {}
        initial_last_key = list(self._files.keys())[-1] if len(self) else None
        if self._auto:
            exporters = self.exporters[:]
            for exporter in exporters:
                self.exporters = [exporter]
                info = dict(info, notebook=self.notebook_name)
                super().add(obj, filename, data, info=info)
                new_last_key = list(self._files.keys())[-1] if len(self) else None
                if new_last_key != initial_last_key:
                    self._replacements[new_last_key] = html
            self.exporters = exporters

    def _generate_html(self, node, substitutions):
        exporter = HTMLExporter()
        exporter.register_preprocessor(Substitute(self.nbversion, substitutions))
        html, _ = exporter.from_notebook_node(node)
        return html

    def _clear_notebook(self, node):
        exporter = NotebookExporter()
        exporter.register_preprocessor(ClearOutputPreprocessor(enabled=True))
        cleared, _ = exporter.from_notebook_node(node)
        return cleared

    def _export_with_html(self):
        """Computes substitutions before using nbconvert with preprocessors"""
        self.export_success = False
        try:
            tstamp = time.strftime(self.timestamp_format, self._timestamp)
            substitutions = {}
            for (basename, ext), entry in self._files.items():
                _, info = entry
                html_key = self._replacements.get((basename, ext), None)
                if html_key is None:
                    continue
                filename = self._format(basename, {'timestamp': tstamp, 'notebook': self.notebook_name})
                fpath = filename + (f'.{ext}' if ext else '')
                info = {'src': fpath, 'mime_type': info['mime_type']}
                if 'mime_type' not in info:
                    pass
                elif info['mime_type'] not in self._tags:
                    pass
                else:
                    basename, ext = os.path.splitext(fpath)
                    truncated = self._truncate_name(basename, ext[1:])
                    link_html = self._format(self._tags[info['mime_type']], {'src': truncated, 'mime_type': info['mime_type'], 'css': ''})
                    substitutions[html_key] = (link_html, truncated)
            node = self._get_notebook_node()
            html = self._generate_html(node, substitutions)
            export_filename = self.snapshot_name
            info = {'file-ext': 'html', 'mime_type': 'text/html', 'notebook': self.notebook_name}
            super().add(filename=export_filename, data=html, info=info)
            cleared = self._clear_notebook(node)
            info = {'file-ext': 'ipynb', 'mime_type': 'text/json', 'notebook': self.notebook_name}
            super().add(filename=export_filename, data=cleared, info=info)
            super().export(timestamp=self._timestamp, info={'notebook': self.notebook_name})
        except Exception:
            self.traceback = traceback.format_exc()
        else:
            self.export_success = True

    def _get_notebook_node(self):
        """Load captured notebook node"""
        size = len(self._notebook_data)
        if size == 0:
            raise Exception('Captured buffer size for notebook node is zero.')
        node = reader.reads(self._notebook_data)
        self.nbversion = reader.get_version(node)
        return node