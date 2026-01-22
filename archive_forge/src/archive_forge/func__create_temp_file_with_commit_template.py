import codecs
import os
import sys
from io import BytesIO, StringIO
from subprocess import call
from . import bedding, cmdline, config, osutils, trace, transport, ui
from .errors import BzrError
from .hooks import Hooks
def _create_temp_file_with_commit_template(infotext, ignoreline=DEFAULT_IGNORE_LINE, start_message=None, tmpdir=None):
    """Create temp file and write commit template in it.

    :param infotext: Text to be displayed at bottom of message for the
        user's reference; currently similar to 'bzr status'.  The text is
        already encoded.

    :param ignoreline:  The separator to use above the infotext.

    :param start_message: The text to place above the separator, if any.
        This will not be removed from the message after the user has edited
        it.  The string is already encoded

    :return:    2-tuple (temp file name, hasinfo)
    """
    import tempfile
    tmp_fileno, msgfilename = tempfile.mkstemp(prefix='bzr_log.', dir=tmpdir, text=True)
    with os.fdopen(tmp_fileno, 'wb') as msgfile:
        if start_message is not None:
            msgfile.write(b'%s\n' % start_message)
        if infotext is not None and infotext != '':
            hasinfo = True
            trailer = b'\n\n%s\n\n%s' % (ignoreline.encode(osutils.get_user_encoding()), infotext)
            msgfile.write(trailer)
        else:
            hasinfo = False
    return (msgfilename, hasinfo)