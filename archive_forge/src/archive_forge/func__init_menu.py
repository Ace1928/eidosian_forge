import functools
import itertools
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import warnings
import zipfile
from hashlib import md5
from xml.etree import ElementTree
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import nltk
def _init_menu(self):
    menubar = Menu(self.top)
    filemenu = Menu(menubar, tearoff=0)
    filemenu.add_command(label='Download', underline=0, command=self._download, accelerator='Return')
    filemenu.add_separator()
    filemenu.add_command(label='Change Server Index', underline=7, command=lambda: self._info_edit('url'))
    filemenu.add_command(label='Change Download Directory', underline=0, command=lambda: self._info_edit('download_dir'))
    filemenu.add_separator()
    filemenu.add_command(label='Show Log', underline=5, command=self._show_log)
    filemenu.add_separator()
    filemenu.add_command(label='Exit', underline=1, command=self.destroy, accelerator='Ctrl-x')
    menubar.add_cascade(label='File', underline=0, menu=filemenu)
    viewmenu = Menu(menubar, tearoff=0)
    for column in self._table.column_names[2:]:
        var = IntVar(self.top)
        assert column not in self._column_vars
        self._column_vars[column] = var
        if column in self.INITIAL_COLUMNS:
            var.set(1)
        viewmenu.add_checkbutton(label=column, underline=0, variable=var, command=self._select_columns)
    menubar.add_cascade(label='View', underline=0, menu=viewmenu)
    sortmenu = Menu(menubar, tearoff=0)
    for column in self._table.column_names[1:]:
        sortmenu.add_command(label='Sort by %s' % column, command=lambda c=column: self._table.sort_by(c, 'ascending'))
    sortmenu.add_separator()
    for column in self._table.column_names[1:]:
        sortmenu.add_command(label='Reverse sort by %s' % column, command=lambda c=column: self._table.sort_by(c, 'descending'))
    menubar.add_cascade(label='Sort', underline=0, menu=sortmenu)
    helpmenu = Menu(menubar, tearoff=0)
    helpmenu.add_command(label='About', underline=0, command=self.about)
    helpmenu.add_command(label='Instructions', underline=0, command=self.help, accelerator='F1')
    menubar.add_cascade(label='Help', underline=0, menu=helpmenu)
    self.top.bind('<F1>', self.help)
    self.top.config(menu=menubar)