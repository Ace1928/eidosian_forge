from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
import sndhdr
from io import BytesIO
from future.backports.email import encoders
from future.backports.email.mime.nonmultipart import MIMENonMultipart
def _whatsnd(data):
    """Try to identify a sound file type.

    sndhdr.what() has a pretty cruddy interface, unfortunately.  This is why
    we re-do it here.  It would be easier to reverse engineer the Unix 'file'
    command and use the standard 'magic' file, as shipped with a modern Unix.
    """
    hdr = data[:512]
    fakefile = BytesIO(hdr)
    for testfn in sndhdr.tests:
        res = testfn(hdr, fakefile)
        if res is not None:
            return _sndhdr_MIMEmap.get(res[0])
    return None