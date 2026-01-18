import datetime as dt
from unittest import TestCase
from unittest.mock import MagicMock
from traitlets import TraitError
from ipywidgets import FileUpload
def test_receive_multiple_files(self):
    uploader = FileUpload(multiple=True)
    message = {'value': [FILE_UPLOAD_FRONTEND_CONTENT, {**FILE_UPLOAD_FRONTEND_CONTENT, **{'name': 'other-file-name.txt'}}]}
    uploader.set_state(message)
    assert len(uploader.value) == 2
    assert uploader.value[0].name == 'file-name.txt'
    assert uploader.value[1].name == 'other-file-name.txt'