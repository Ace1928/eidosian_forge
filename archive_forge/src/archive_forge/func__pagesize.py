import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def _pagesize(self):
    """:return: The number of rows that makes up one page"""
    return int(self.index('@0,1000000')) - int(self.index('@0,0'))