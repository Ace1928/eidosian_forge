import io
import math
import os
import typing
import weakref
def updateRect(self, x):
    if self.rect is None:
        if len(x) == 2:
            self.rect = fitz.Rect(x, x)
        else:
            self.rect = fitz.Rect(x)
    elif len(x) == 2:
        x = fitz.Point(x)
        self.rect.x0 = min(self.rect.x0, x.x)
        self.rect.y0 = min(self.rect.y0, x.y)
        self.rect.x1 = max(self.rect.x1, x.x)
        self.rect.y1 = max(self.rect.y1, x.y)
    else:
        x = fitz.Rect(x)
        self.rect.x0 = min(self.rect.x0, x.x0)
        self.rect.y0 = min(self.rect.y0, x.y0)
        self.rect.x1 = max(self.rect.x1, x.x1)
        self.rect.y1 = max(self.rect.y1, x.y1)