import os
import sys
import tempfile
from rdkit import Chem
def GetPNG(self, h=None, w=None, preDelay=0):
    try:
        import Image
    except ImportError:
        from PIL import Image
    import time
    if preDelay > 0:
        time.sleep(preDelay)
    fd = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fd.close()
    self.server.do('png %s' % fd.name)
    time.sleep(0.2)
    for i in range(10):
        try:
            img = Image.open(fd.name)
            break
        except IOError:
            time.sleep(0.1)
    try:
        os.unlink(fd.name)
    except (OSError, PermissionError):
        pass
    fd = None
    if h is not None or w is not None:
        sz = img.size
        if h is None:
            h = sz[1]
        if w is None:
            w = sz[0]
        if h < sz[1]:
            frac = float(h) / sz[1]
            w *= frac
            w = int(w)
            img = img.resize((w, h), True)
        elif w < sz[0]:
            frac = float(w) / sz[0]
            h *= frac
            h = int(h)
            img = img.resize((w, h), True)
    return img