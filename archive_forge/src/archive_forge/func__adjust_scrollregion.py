from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def _adjust_scrollregion(self):
    """
        Adjust the scrollregion of this scroll-watcher's ``Canvas`` to
        include the bounding boxes of all of its children.
        """
    bbox = self.bbox()
    canvas = self.canvas()
    scrollregion = [int(n) for n in canvas['scrollregion'].split()]
    if len(scrollregion) != 4:
        return
    if bbox[0] < scrollregion[0] or bbox[1] < scrollregion[1] or bbox[2] > scrollregion[2] or (bbox[3] > scrollregion[3]):
        scrollregion = '%d %d %d %d' % (min(bbox[0], scrollregion[0]), min(bbox[1], scrollregion[1]), max(bbox[2], scrollregion[2]), max(bbox[3], scrollregion[3]))
        canvas['scrollregion'] = scrollregion