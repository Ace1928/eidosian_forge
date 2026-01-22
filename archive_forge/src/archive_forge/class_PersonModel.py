from __future__ import print_function
import os
import sys
import PySide2.QtQml
from PySide2.QtCore import QAbstractListModel, Qt, QUrl, QByteArray
from PySide2.QtGui import QGuiApplication
from PySide2.QtQuick import QQuickView
class PersonModel(QAbstractListModel):
    MyRole = Qt.UserRole + 1

    def __init__(self, parent=None):
        QAbstractListModel.__init__(self, parent)
        self._data = []

    def roleNames(self):
        roles = {PersonModel.MyRole: QByteArray(b'modelData'), Qt.DisplayRole: QByteArray(b'display')}
        return roles

    def rowCount(self, index):
        return len(self._data)

    def data(self, index, role):
        d = self._data[index.row()]
        if role == Qt.DisplayRole:
            return d['name']
        elif role == Qt.DecorationRole:
            return Qt.black
        elif role == PersonModel.MyRole:
            return d['myrole']
        return None

    def populate(self):
        self._data.append({'name': 'Qt', 'myrole': 'role1'})
        self._data.append({'name': 'PySide', 'myrole': 'role2'})