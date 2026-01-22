from __future__ import absolute_import  #, unicode_literals
def allmarkers(self):
    children = self.prepended_children
    return [m for c in children for m in c.allmarkers()] + self.markers