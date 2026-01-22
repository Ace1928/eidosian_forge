import os
from base64 import b64decode
from traitlets import Bool, Unicode
from .base import Preprocessor
class ExtractAttachmentsPreprocessor(Preprocessor):
    """
    Extracts attachments from all (markdown and raw) cells in a notebook.
    The extracted attachments are stored in a directory ('attachments' by default).
    https://nbformat.readthedocs.io/en/latest/format_description.html#cell-attachments
    """
    attachments_directory_template = Unicode('{notebook_name}_attachments', help='Directory to place attachments if use_separate_dir is True').tag(config=True)
    use_separate_dir = Bool(False, help='Whether to use output_files_dir (which ExtractOutput also uses) or create a separate directory for attachments').tag(config=True)

    def __init__(self, **kw):
        """
        Public constructor
        """
        super().__init__(**kw)
        self.path_name = ''
        self.resources_item_key = 'attachments'

    def preprocess(self, nb, resources):
        """
        Determine some settings and apply preprocessor to notebook
        """
        if self.use_separate_dir:
            self.path_name = self.attachments_directory_template.format(notebook_name=resources['unique_key'])
            resources['attachment_files_dir'] = self.path_name
            resources['attachments'] = {}
            self.resources_item_key = 'attachments'
        else:
            self.path_name = resources['output_files_dir']
            self.resources_item_key = 'outputs'
        if not isinstance(resources[self.resources_item_key], dict):
            resources[self.resources_item_key] = {}
        nb, resources = super().preprocess(nb, resources)
        return (nb, resources)

    def preprocess_cell(self, cell, resources, index):
        """
        Extract attachments to individual files and
        change references to them.
        E.g.
        '![image.png](attachment:021fdd80.png)'
        becomes
        '![image.png]({path_name}/021fdd80.png)'
        Assumes self.path_name and self.resources_item_key is set properly (usually in preprocess).
        """
        if 'attachments' in cell:
            for fname in cell.attachments:
                self.log.debug('Encountered attachment %s', fname)
                for mimetype in cell.attachments[fname]:
                    data = cell.attachments[fname][mimetype].encode('utf-8')
                    decoded = b64decode(data)
                    break
                new_filename = os.path.join(self.path_name, fname)
                resources[self.resources_item_key][new_filename] = decoded
                if os.path.sep != '/':
                    new_filename = new_filename.replace(os.path.sep, '/')
                cell.source = cell.source.replace('attachment:' + fname, new_filename)
        return (cell, resources)