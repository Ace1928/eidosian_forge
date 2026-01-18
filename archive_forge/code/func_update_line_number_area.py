from PySide2.QtCore import Slot, Qt, QRect, QSize
from PySide2.QtGui import QColor, QPainter, QTextFormat
from PySide2.QtWidgets import QPlainTextEdit, QWidget, QTextEdit
@Slot()
def update_line_number_area(self, rect, dy):
    if dy:
        self.line_number_area.scroll(0, dy)
    else:
        width = self.line_number_area.width()
        self.line_number_area.update(0, rect.y(), width, rect.height())
    if rect.contains(self.viewport().rect()):
        self.update_line_number_area_width(0)