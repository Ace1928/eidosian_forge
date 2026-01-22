import json
import os
import sys
from binascii import a2b_base64
from mimetypes import guess_extension
from textwrap import dedent
from traitlets import Set, Unicode
from .base import Preprocessor
class ExtractOutputPreprocessor(Preprocessor):
    """
    Extracts all of the outputs from the notebook file.  The extracted
    outputs are returned in the 'resources' dictionary.
    """
    output_filename_template = Unicode('{unique_key}_{cell_index}_{index}{extension}').tag(config=True)
    extract_output_types = Set({'image/png', 'image/jpeg', 'image/svg+xml', 'application/pdf'}).tag(config=True)

    def preprocess_cell(self, cell, resources, cell_index):
        """
        Apply a transformation on each cell,

        Parameters
        ----------
        cell : NotebookNode cell
            Notebook cell being processed
        resources : dictionary
            Additional resources used in the conversion process.  Allows
            preprocessors to pass variables into the Jinja engine.
        cell_index : int
            Index of the cell being processed (see base.py)
        """
        unique_key = resources.get('unique_key', 'output')
        output_files_dir = resources.get('output_files_dir', None)
        if not isinstance(resources['outputs'], dict):
            resources['outputs'] = {}
        for index, out in enumerate(cell.get('outputs', [])):
            if out.output_type not in {'display_data', 'execute_result'}:
                continue
            if 'text/html' in out.data:
                out['data']['text/html'] = dedent(out['data']['text/html'])
            for mime_type in self.extract_output_types:
                if mime_type in out.data:
                    data = out.data[mime_type]
                    if mime_type in {'image/png', 'image/jpeg', 'application/pdf'}:
                        data = a2b_base64(data)
                    elif mime_type == 'application/json' or not isinstance(data, str):
                        if isinstance(data, bytes):
                            data = data.decode('utf-8')
                        data = platform_utf_8_encode(json.dumps(data))
                    else:
                        data = platform_utf_8_encode(data)
                    ext = guess_extension_without_jpe(mime_type)
                    if ext is None:
                        ext = '.' + mime_type.rsplit('/')[-1]
                    if out.metadata.get('filename', ''):
                        filename = out.metadata['filename']
                        if not filename.endswith(ext):
                            filename += ext
                    else:
                        filename = self.output_filename_template.format(unique_key=unique_key, cell_index=cell_index, index=index, extension=ext)
                    if output_files_dir is not None:
                        filename = os.path.join(output_files_dir, filename)
                    out.metadata.setdefault('filenames', {})
                    out.metadata['filenames'][mime_type] = filename
                    if filename in resources['outputs']:
                        msg = f'Your outputs have filename metadata associated with them. Nbconvert saves these outputs to external files using this filename metadata. Filenames need to be unique across the notebook, or images will be overwritten. The filename {filename} is associated with more than one output. The second output associated with this filename is in cell {cell_index}.'
                        raise ValueError(msg)
                    resources['outputs'][filename] = data
        return (cell, resources)