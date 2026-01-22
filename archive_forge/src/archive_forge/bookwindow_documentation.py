from __future__ import print_function, absolute_import
from PySide2.QtWidgets import (QAction, QAbstractItemView, qApp, QDataWidgetMapper,
from PySide2.QtGui import QKeySequence
from PySide2.QtSql import (QSqlRelation, QSqlRelationalTableModel, QSqlTableModel,
from PySide2.QtCore import QAbstractItemModel, QObject, QSize, Qt, Slot
import createdb
from ui_bookwindow import Ui_BookWindow
from bookdelegate import BookDelegate
A window to show the books available