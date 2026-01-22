from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import subprocess
import tempfile
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def OnlineEdit(text):
    """Edit will edit the provided text.

  Args:
    text: The initial text blob to provide for editing.

  Returns:
    The edited text blob.

  Raises:
    NoSaveException: If the user did not save the temporary file.
    EditorException: If the process running the editor has a
        problem.
  """
    fname = tempfile.NamedTemporaryFile(suffix='.txt').name
    files.WriteFileContents(fname, text)
    start_mtime = FileModifiedTime(fname)
    if platforms.OperatingSystem.Current() is platforms.OperatingSystem.WINDOWS:
        try:
            SubprocessCheckCall([fname], shell=True)
        except subprocess.CalledProcessError as error:
            raise EditorException('Your editor exited with return code {0}; please try again.'.format(error.returncode))
    else:
        try:
            editor = encoding.GetEncodedValue(os.environ, 'EDITOR', 'vi')
            SubprocessCheckCall('{editor} {file}'.format(editor=editor, file=fname), shell=True)
        except subprocess.CalledProcessError as error:
            raise EditorException('Your editor exited with return code {0}; please try again. You may set the EDITOR environment to use a different text editor.'.format(error.returncode))
    end_mtime = FileModifiedTime(fname)
    if start_mtime == end_mtime:
        raise NoSaveException('edit aborted by user')
    return files.ReadFileContents(fname)