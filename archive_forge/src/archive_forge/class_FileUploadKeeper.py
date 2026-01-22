import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class FileUploadKeeper(FancyValidator):
    """
    Takes two inputs (a dictionary with keys ``static`` and
    ``upload``) and converts them into one value on the Python side (a
    dictionary with ``filename`` and ``content`` keys).  The upload
    takes priority over the static value.  The filename may be None if
    it can't be discovered.

    Handles uploads of both text and ``cgi.FieldStorage`` upload
    values.

    This is basically for use when you have an upload field, and you
    want to keep the upload around even if the rest of the form
    submission fails.  When converting *back* to the form submission,
    there may be extra values ``'original_filename'`` and
    ``'original_content'``, which may want to use in your form to show
    the user you still have their content around.

    To use this, make sure you are using variabledecode, then use
    something like::

      <input type="file" name="myfield.upload">
      <input type="hidden" name="myfield.static">

    Then in your scheme::

      class MyScheme(Scheme):
          myfield = FileUploadKeeper()

    Note that big file uploads mean big hidden fields, and lots of
    bytes passed back and forth in the case of an error.
    """
    upload_key = 'upload'
    static_key = 'static'

    def _convert_to_python(self, value, state):
        upload = value.get(self.upload_key)
        static = value.get(self.static_key, '').strip()
        filename = content = None
        if isinstance(upload, cgi.FieldStorage):
            filename = upload.filename
            content = upload.value
        elif isinstance(upload, str) and upload:
            filename = None
            content = upload
        if not content and static:
            filename, content = static.split(None, 1)
            filename = '' if filename == '-' else filename.decode('base64')
            content = content.decode('base64')
        return {'filename': filename, 'content': content}

    def _convert_from_python(self, value, state):
        filename = value.get('filename', '')
        content = value.get('content', '')
        if filename or content:
            result = self.pack_content(filename, content)
            return {self.upload_key: '', self.static_key: result, 'original_filename': filename, 'original_content': content}
        else:
            return {self.upload_key: '', self.static_key: ''}

    def pack_content(self, filename, content):
        enc_filename = self.base64encode(filename) or '-'
        enc_content = (content or '').encode('base64')
        result = '%s %s' % (enc_filename, enc_content)
        return result